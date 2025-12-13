"""
Freeciv Gymnasium Environment using fcgym C library.

This uses CFFI to bind to the fcgym library which provides direct access
to the Freeciv game engine without network protocols.

Note: fcgym uses global Freeciv state. Multiple env instances can be created
sequentially (create, use, close, create new), but only one game runs at a time.
The library is initialized once per process and cleaned up automatically on exit.

Action Space:
- Discrete(MAX_LEGAL) with state-dependent legal action masking
- Each step returns action_mask (fixed size) and legal_actions (fixed size, padded) in info
- legal_actions[i] = [action_type, actor_slot, target, sub_target]

Observation Space:
- Dict with: global, map, units, cities, players arrays
- units/cities only include OUR controllable entities (stable slots)
- Enemy visibility via map channels (ownership_enemy, has_unit on tiles)
"""

import atexit
import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Dict, Any, Tuple, List
from cffi import FFI
from enum import IntEnum


# Constants
MAX_LEGAL_ACTIONS = 1024  # Configurable cap on legal actions per step
MAX_UNITS = 256
MAX_CITIES = 64
MAX_PLAYERS = 8
MAP_CHANNELS = 9  # visibility, terrain, road, irrigation, mine, ownership_self, ownership_enemy, city, unit_visible

# Module-level library state (shared across all env instances in this process)
_lib_handle = None
_library_initialized = False


def _shutdown_library():
    """Shutdown the fcgym library. Called automatically on process exit."""
    global _library_initialized, _lib_handle
    if _library_initialized and _lib_handle is not None:
        _lib_handle.fcgym_shutdown()
        _library_initialized = False
        _lib_handle = None


# Register shutdown handler for clean process exit
atexit.register(_shutdown_library)


def shutdown_library():
    """Explicitly shutdown the fcgym library.

    Normally not needed - the library shuts down automatically on process exit.
    Only call this if you need to free resources before the process ends.
    After calling this, no FreecivGymEnv instances can be used until a new
    process is started.
    """
    _shutdown_library()


class FcActionType(IntEnum):
    """Action types matching fcgym.h FcActionType enum exactly."""
    UNIT_MOVE = 0
    UNIT_ATTACK = 1
    UNIT_FORTIFY = 2
    UNIT_BUILD_CITY = 3
    UNIT_BUILD_ROAD = 4
    UNIT_BUILD_IRRIGATION = 5
    UNIT_BUILD_MINE = 6
    UNIT_DISBAND = 7
    CITY_BUILD = 8
    CITY_BUY = 9
    RESEARCH_SET = 10
    END_TURN = 11
    NOOP = 12


# CFFI setup
ffi = FFI()

# C definitions for fcgym structs and functions
# IMPORTANT: These must match fcgym.h exactly for memory layout
ffi.cdef("""
    typedef struct {
        const char *ruleset;
        int map_xsize;
        int map_ysize;
        int num_ai_players;
        int ai_skill_level;
        unsigned int seed;
        bool fog_of_war;
    } FcGameConfig;

    typedef struct {
        int terrain;
        int owner;
        bool has_city;
        bool has_unit;
        bool visible;
        bool explored;
        int8_t extras;
    } FcTileObs;

    typedef struct {
        int id;
        int type;
        int owner;
        int tile_index;
        int hp;
        int max_hp;
        int moves_left;
        int veteran_level;
        bool fortified;
    } FcUnitObs;

    typedef struct {
        int id;
        int owner;
        int tile_index;
        int size;
        int food_stock;
        int shield_stock;
        int producing_type;
        bool producing_is_unit;
        int turns_to_complete;
    } FcCityObs;

    typedef struct {
        int index;
        bool is_alive;
        bool is_ai;
        int gold;
        int tax_rate;
        int science_rate;
        int luxury_rate;
        int researching;
        int research_bulbs;
        int num_cities;
        int num_units;
        int score;
    } FcPlayerObs;

    typedef struct {
        int map_xsize;
        int map_ysize;

        int turn;
        int year;
        int phase;
        int current_player;
        int controlled_player;

        FcTileObs *tiles;
        int num_tiles;

        FcUnitObs *units;
        int num_units;
        int max_units;

        FcCityObs *cities;
        int num_cities;
        int max_cities;

        FcPlayerObs *players;
        int num_players;

        bool game_over;
        int winner;
    } FcObservation;

    typedef struct {
        int unit_id;
        bool can_move[8];
        int attackable_tiles[8];
        int num_attackable_tiles;
        bool can_fortify;
        bool can_build_city;
        bool can_build_road;
        bool can_build_irrigation;
        bool can_build_mine;
        bool can_disband;
    } FcUnitActions;

    typedef struct {
        int city_id;
        int *buildable_units;
        int num_buildable_units;
        int *buildable_buildings;
        int num_buildable_buildings;
        bool can_buy;
    } FcCityActions;

    typedef struct {
        FcUnitActions *unit_actions;
        int num_unit_actions;

        FcCityActions *city_actions;
        int num_city_actions;

        int *researchable_techs;
        int num_researchable_techs;

        bool can_end_turn;
    } FcValidActions;

    typedef struct {
        int type;
        int actor_id;
        int target_id;
        int sub_target;
    } FcAction;

    typedef struct {
        float reward;
        bool done;
        bool truncated;
        const char *info;
    } FcStepResult;

    int fcgym_init(void);
    void fcgym_shutdown(void);
    int fcgym_new_game(FcGameConfig *config);
    void fcgym_get_observation(FcObservation *obs);
    void fcgym_free_observation(FcObservation *obs);
    void fcgym_get_valid_actions(FcValidActions *actions);
    void fcgym_free_valid_actions(FcValidActions *actions);
    FcStepResult fcgym_step(FcAction *action);
    const char* fcgym_unit_type_name(int unit_type);
    const char* fcgym_tech_name(int tech_id);
""")


def _find_library():
    """Find the fcgym library (prefer shared library for CFFI)."""
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Prefer shared library (.so) for CFFI dynamic loading
    candidates = [
        os.path.join(script_dir, "libfcgym.so"),
        os.path.join(script_dir, "..", "build", "fcgym", "libfcgym.so"),
        os.path.join(script_dir, "libfcgym.a"),  # Fallback (will error but with message)
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    return None


class FreecivGymEnv(gym.Env):
    """
    Gymnasium environment for Freeciv using fcgym C library.

    Observation space is a Dict with:
    - global: (G,) global game state
    - map: (C, H, W) map channels
    - units: (MAX_UNITS, U) unit features
    - unit_mask: (MAX_UNITS,) valid unit mask
    - cities: (MAX_CITIES, C) city features
    - city_mask: (MAX_CITIES,) valid city mask
    - players: (MAX_PLAYERS, P) player features
    - player_mask: (MAX_PLAYERS,) valid player mask

    Action space is Discrete(MAX_LEGAL) with:
    - action_mask in info
    - legal_actions array encoding [type, actor_slot, target, sub_target]
    """

    metadata = {"render_modes": ["human", "ansi"]}

    def __init__(
        self,
        ruleset: str = "civ2civ3",
        map_width: int = 40,
        map_height: int = 40,
        num_ai_players: int = 2,
        ai_skill_level: int = 3,
        seed: Optional[int] = None,
        fog_of_war: bool = True,
        max_legal_actions: int = MAX_LEGAL_ACTIONS,
        render_mode: Optional[str] = None,
    ):
        super().__init__()

        self.ruleset = ruleset
        self.map_width = map_width
        self.map_height = map_height
        self.num_ai_players = num_ai_players
        self.ai_skill_level = ai_skill_level
        self.seed = seed
        self.fog_of_war = fog_of_war
        self.max_legal_actions = max_legal_actions
        self.render_mode = render_mode

        # Load library
        self._lib = None
        self._initialized = False

        # Slot mappings (engine_id -> slot_index)
        self._unit_id_to_slot: Dict[int, int] = {}
        self._city_id_to_slot: Dict[int, int] = {}
        self._slot_to_unit_id: Dict[int, int] = {}
        self._slot_to_city_id: Dict[int, int] = {}

        # Current legal actions cache
        self._legal_actions = np.zeros((max_legal_actions, 4), dtype=np.int32)
        self._action_mask = np.zeros(max_legal_actions, dtype=np.float32)
        self._num_legal_actions = 0

        # Define spaces
        self._define_spaces()

    def _define_spaces(self):
        """Define observation and action spaces."""
        # Global features: turn, year, phase, controlled_player, etc.
        global_size = 10

        # Unit features: x, y, hp, moves_left, veteran, type_id, owner_relative, activity
        unit_features = 10

        # City features: x, y, size, food_stock, shield_stock, producing_type, turns_to_complete, owner_relative
        city_features = 10

        # Player features: gold, tax, science, luxury, research_bulbs, num_cities, num_units, score, is_alive, is_ai
        player_features = 12

        self.observation_space = spaces.Dict({
            "global": spaces.Box(low=-np.inf, high=np.inf, shape=(global_size,), dtype=np.float32),
            "map": spaces.Box(low=0, high=255, shape=(MAP_CHANNELS, self.map_height, self.map_width), dtype=np.uint8),
            "units": spaces.Box(low=-np.inf, high=np.inf, shape=(MAX_UNITS, unit_features), dtype=np.float32),
            "unit_mask": spaces.Box(low=0, high=1, shape=(MAX_UNITS,), dtype=np.float32),
            "cities": spaces.Box(low=-np.inf, high=np.inf, shape=(MAX_CITIES, city_features), dtype=np.float32),
            "city_mask": spaces.Box(low=0, high=1, shape=(MAX_CITIES,), dtype=np.float32),
            "players": spaces.Box(low=-np.inf, high=np.inf, shape=(MAX_PLAYERS, player_features), dtype=np.float32),
            "player_mask": spaces.Box(low=0, high=1, shape=(MAX_PLAYERS,), dtype=np.float32),
        })

        self.action_space = spaces.Discrete(self.max_legal_actions)

    def _load_library(self):
        """Load the fcgym library using CFFI."""
        global _lib_handle

        # Use module-level handle if already loaded
        if _lib_handle is not None:
            self._lib = _lib_handle
            return

        if self._lib is not None:
            return

        lib_path = _find_library()
        if lib_path is None:
            raise RuntimeError(
                "Could not find fcgym library. Build it with: cd fcgym && make"
            )

        # Note: CFFI with static libraries requires linking at build time
        # For dynamic loading, we need a shared library (.so)
        if lib_path.endswith('.a'):
            raise RuntimeError(
                f"Found static library {lib_path}, but CFFI needs a shared library (.so). "
                "Add 'shared: true' to the meson build or build manually with -shared flag."
            )

        self._lib = ffi.dlopen(lib_path)
        _lib_handle = self._lib

    def _init_fcgym(self):
        """Initialize fcgym library."""
        global _library_initialized

        # Library already initialized in this process - just update instance state
        if _library_initialized:
            self._load_library()  # Ensure self._lib is set
            self._initialized = True
            return

        self._load_library()
        result = self._lib.fcgym_init()
        if result != 0:
            raise RuntimeError("Failed to initialize fcgym")
        _library_initialized = True
        self._initialized = True

    def _update_slot_mappings(self, obs):
        """Update stable slot mappings from observation.

        Only our own controllable units/cities are slotted to ensure stability.
        Enemy units appearing/disappearing won't shift our slots.
        Slots are assigned by sorting engine IDs for consistency.
        """
        controlled = obs.controlled_player

        # Get only OUR unit IDs and sort them
        our_unit_ids = sorted([
            obs.units[i].id
            for i in range(obs.num_units)
            if obs.units[i].owner == controlled
        ])
        self._unit_id_to_slot.clear()
        self._slot_to_unit_id.clear()
        for slot, uid in enumerate(our_unit_ids):
            self._unit_id_to_slot[uid] = slot
            self._slot_to_unit_id[slot] = uid

        # Get only OUR city IDs and sort them
        our_city_ids = sorted([
            obs.cities[i].id
            for i in range(obs.num_cities)
            if obs.cities[i].owner == controlled
        ])
        self._city_id_to_slot.clear()
        self._slot_to_city_id.clear()
        for slot, cid in enumerate(our_city_ids):
            self._city_id_to_slot[cid] = slot
            self._slot_to_city_id[slot] = cid

    def _build_legal_actions(self, valid_actions):
        """Build legal actions array from FcValidActions.

        Each legal action is encoded as [type, actor_slot, target, sub_target].
        """
        self._legal_actions.fill(0)
        self._action_mask.fill(0)
        idx = 0

        # END_TURN
        if valid_actions.can_end_turn and idx < self.max_legal_actions:
            self._legal_actions[idx] = [FcActionType.END_TURN, 0, 0, 0]
            self._action_mask[idx] = 1.0
            idx += 1

        # Unit actions
        for i in range(valid_actions.num_unit_actions):
            ua = valid_actions.unit_actions[i]
            unit_id = ua.unit_id
            if unit_id not in self._unit_id_to_slot:
                continue
            slot = self._unit_id_to_slot[unit_id]

            # Movement in 8 directions (non-combat moves only)
            for d in range(8):
                if ua.can_move[d] and idx < self.max_legal_actions:
                    self._legal_actions[idx] = [FcActionType.UNIT_MOVE, slot, 0, d]
                    self._action_mask[idx] = 1.0
                    idx += 1

            # Attack actions - enumerate from attackable_tiles array
            for a in range(ua.num_attackable_tiles):
                if idx < self.max_legal_actions:
                    target_tile = ua.attackable_tiles[a]
                    self._legal_actions[idx] = [FcActionType.UNIT_ATTACK, slot, target_tile, 0]
                    self._action_mask[idx] = 1.0
                    idx += 1

            # Fortify
            if ua.can_fortify and idx < self.max_legal_actions:
                self._legal_actions[idx] = [FcActionType.UNIT_FORTIFY, slot, 0, 0]
                self._action_mask[idx] = 1.0
                idx += 1

            # Build city
            if ua.can_build_city and idx < self.max_legal_actions:
                self._legal_actions[idx] = [FcActionType.UNIT_BUILD_CITY, slot, 0, 0]
                self._action_mask[idx] = 1.0
                idx += 1

            # Build road
            if ua.can_build_road and idx < self.max_legal_actions:
                self._legal_actions[idx] = [FcActionType.UNIT_BUILD_ROAD, slot, 0, -1]
                self._action_mask[idx] = 1.0
                idx += 1

            # Build irrigation
            if ua.can_build_irrigation and idx < self.max_legal_actions:
                self._legal_actions[idx] = [FcActionType.UNIT_BUILD_IRRIGATION, slot, 0, -1]
                self._action_mask[idx] = 1.0
                idx += 1

            # Build mine
            if ua.can_build_mine and idx < self.max_legal_actions:
                self._legal_actions[idx] = [FcActionType.UNIT_BUILD_MINE, slot, 0, -1]
                self._action_mask[idx] = 1.0
                idx += 1

            # Disband
            if ua.can_disband and idx < self.max_legal_actions:
                self._legal_actions[idx] = [FcActionType.UNIT_DISBAND, slot, 0, 0]
                self._action_mask[idx] = 1.0
                idx += 1

        # City actions
        for i in range(valid_actions.num_city_actions):
            ca = valid_actions.city_actions[i]
            city_id = ca.city_id
            if city_id not in self._city_id_to_slot:
                continue
            slot = self._city_id_to_slot[city_id]

            # Set production (units)
            for j in range(ca.num_buildable_units):
                if idx < self.max_legal_actions:
                    unit_type = ca.buildable_units[j]
                    self._legal_actions[idx] = [FcActionType.CITY_BUILD, slot, unit_type, 0]
                    self._action_mask[idx] = 1.0
                    idx += 1

            # Set production (buildings)
            for j in range(ca.num_buildable_buildings):
                if idx < self.max_legal_actions:
                    bldg_type = ca.buildable_buildings[j]
                    self._legal_actions[idx] = [FcActionType.CITY_BUILD, slot, bldg_type, 1]
                    self._action_mask[idx] = 1.0
                    idx += 1

            # Buy
            if ca.can_buy and idx < self.max_legal_actions:
                self._legal_actions[idx] = [FcActionType.CITY_BUY, slot, 0, 0]
                self._action_mask[idx] = 1.0
                idx += 1

        # Research
        for i in range(valid_actions.num_researchable_techs):
            if idx < self.max_legal_actions:
                tech_id = valid_actions.researchable_techs[i]
                self._legal_actions[idx] = [FcActionType.RESEARCH_SET, 0, tech_id, 0]
                self._action_mask[idx] = 1.0
                idx += 1

        self._num_legal_actions = idx

    def _build_observation(self, obs) -> Dict[str, np.ndarray]:
        """Build observation dict from FcObservation."""
        # Global features
        global_obs = np.zeros(10, dtype=np.float32)
        global_obs[0] = obs.turn
        global_obs[1] = obs.controlled_player
        global_obs[2] = obs.num_players
        global_obs[3] = obs.map_xsize
        global_obs[4] = obs.map_ysize
        global_obs[5] = 1.0 if obs.game_over else 0.0
        global_obs[6] = obs.year
        global_obs[7] = obs.phase
        global_obs[8] = obs.current_player
        global_obs[9] = obs.winner if obs.game_over else -1

        # Map observation from tiles
        # Channels:
        #   0=visibility, 1=terrain, 2=road, 3=irrigation, 4=mine,
        #   5=ownership_self, 6=ownership_enemy, 7=city, 8=unit_visible
        map_obs = np.zeros((MAP_CHANNELS, self.map_height, self.map_width), dtype=np.uint8)
        controlled = obs.controlled_player

        for i in range(obs.num_tiles):
            tile = obs.tiles[i]
            x = i % obs.map_xsize
            y = i // obs.map_xsize

            # Skip if out of our observation bounds
            if x >= self.map_width or y >= self.map_height:
                continue

            # Channel 0: visibility (255=visible, 128=explored, 0=unknown)
            if tile.visible:
                map_obs[0, y, x] = 255
            elif tile.explored:
                map_obs[0, y, x] = 128
            # else: stays 0 (unknown)

            # Only fill other channels if tile has been explored
            # (C side sets terrain=-1 for unexplored, which becomes 255 as uint8)
            if tile.explored:
                # Channel 1: terrain type (only if explored)
                map_obs[1, y, x] = tile.terrain if tile.terrain >= 0 else 0

                # Channel 2-4: extras (road, irrigation, mine from extras bitfield)
                if tile.extras & 0x01:
                    map_obs[2, y, x] = 255  # road
                if tile.extras & 0x02:
                    map_obs[3, y, x] = 255  # irrigation
                if tile.extras & 0x04:
                    map_obs[4, y, x] = 255  # mine

                # Channel 5-6: ownership
                if tile.owner >= 0:
                    if tile.owner == controlled:
                        map_obs[5, y, x] = 255  # ownership_self
                    else:
                        map_obs[6, y, x] = 255  # ownership_enemy

                # Channel 7: city presence
                if tile.has_city:
                    map_obs[7, y, x] = 255

                # Channel 8: unit presence (only set when currently visible)
                if tile.has_unit:
                    map_obs[8, y, x] = 255

        # Units
        unit_obs = np.zeros((MAX_UNITS, 10), dtype=np.float32)
        unit_mask = np.zeros(MAX_UNITS, dtype=np.float32)

        for i in range(min(obs.num_units, MAX_UNITS)):
            u = obs.units[i]
            if u.id in self._unit_id_to_slot:
                slot = self._unit_id_to_slot[u.id]
                if slot < MAX_UNITS:
                    # x, y from tile_index
                    x = u.tile_index % obs.map_xsize
                    y = u.tile_index // obs.map_xsize
                    # Owner relative encoding: 0=self, 1=enemy, 2=neutral
                    owner_rel = 0 if u.owner == controlled else 1

                    unit_obs[slot] = [
                        x / obs.map_xsize,
                        y / obs.map_ysize,
                        u.hp / float(u.max_hp) if u.max_hp > 0 else 0.0,
                        u.moves_left / 10.0,
                        u.veteran_level,
                        u.type,
                        owner_rel,
                        1.0 if u.fortified else 0.0,
                        0.0,  # activity
                        u.id,  # For debugging
                    ]
                    unit_mask[slot] = 1.0

        # Cities
        city_obs = np.zeros((MAX_CITIES, 10), dtype=np.float32)
        city_mask = np.zeros(MAX_CITIES, dtype=np.float32)

        for i in range(min(obs.num_cities, MAX_CITIES)):
            c = obs.cities[i]
            if c.id in self._city_id_to_slot:
                slot = self._city_id_to_slot[c.id]
                if slot < MAX_CITIES:
                    x = c.tile_index % obs.map_xsize
                    y = c.tile_index // obs.map_xsize
                    owner_rel = 0 if c.owner == controlled else 1

                    city_obs[slot] = [
                        x / obs.map_xsize,
                        y / obs.map_ysize,
                        c.size / 30.0,
                        c.food_stock / 100.0,
                        c.shield_stock / 100.0,
                        c.producing_type,
                        1.0 if c.producing_is_unit else 0.0,
                        c.turns_to_complete / 50.0,
                        owner_rel,
                        c.id,
                    ]
                    city_mask[slot] = 1.0

        # Players
        player_obs = np.zeros((MAX_PLAYERS, 12), dtype=np.float32)
        player_mask = np.zeros(MAX_PLAYERS, dtype=np.float32)

        for i in range(min(obs.num_players, MAX_PLAYERS)):
            p = obs.players[i]
            player_obs[i] = [
                p.gold / 1000.0,
                p.tax_rate / 100.0,
                p.science_rate / 100.0,
                p.luxury_rate / 100.0,
                p.research_bulbs / 100.0,
                p.num_cities / 20.0,
                p.num_units / 50.0,
                p.score / 1000.0,
                1.0 if p.is_alive else 0.0,
                1.0 if p.is_ai else 0.0,
                p.researching,
                p.index,
            ]
            player_mask[i] = 1.0 if p.is_alive else 0.0

        return {
            "global": global_obs,
            "map": map_obs,
            "units": unit_obs,
            "unit_mask": unit_mask,
            "cities": city_obs,
            "city_mask": city_mask,
            "players": player_obs,
            "player_mask": player_mask,
        }

    def _decode_action(self, action_idx: int) -> Tuple[int, int, int, int]:
        """Decode action index to (type, actor_id, target_id, sub_target)."""
        if action_idx < 0 or action_idx >= self._num_legal_actions:
            # Invalid action - return NOOP
            return (FcActionType.NOOP, 0, 0, 0)

        action = self._legal_actions[action_idx]
        action_type = int(action[0])
        actor_slot = int(action[1])
        target = int(action[2])
        sub_target = int(action[3])

        # Convert slot back to engine ID
        if action_type in (FcActionType.UNIT_MOVE, FcActionType.UNIT_ATTACK,
                           FcActionType.UNIT_FORTIFY, FcActionType.UNIT_BUILD_CITY,
                           FcActionType.UNIT_BUILD_ROAD, FcActionType.UNIT_BUILD_IRRIGATION,
                           FcActionType.UNIT_BUILD_MINE, FcActionType.UNIT_DISBAND):
            if actor_slot in self._slot_to_unit_id:
                actor_id = self._slot_to_unit_id[actor_slot]
            else:
                return (FcActionType.NOOP, 0, 0, 0)
        elif action_type in (FcActionType.CITY_BUILD, FcActionType.CITY_BUY):
            if actor_slot in self._slot_to_city_id:
                actor_id = self._slot_to_city_id[actor_slot]
            else:
                return (FcActionType.NOOP, 0, 0, 0)
        else:
            actor_id = 0

        return (action_type, actor_id, target, sub_target)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Reset the environment for a new episode."""
        super().reset(seed=seed)

        # Initialize library if needed
        self._init_fcgym()

        # Create game config
        config = ffi.new("FcGameConfig *")
        # CFFI requires keeping a reference to the string for char*
        self._ruleset_buf = ffi.new("char[]", self.ruleset.encode('utf-8'))
        config.ruleset = self._ruleset_buf
        config.map_xsize = self.map_width
        config.map_ysize = self.map_height
        config.num_ai_players = self.num_ai_players
        config.ai_skill_level = self.ai_skill_level
        config.seed = seed if seed is not None else (self.seed if self.seed is not None else 0)
        config.fog_of_war = self.fog_of_war

        result = self._lib.fcgym_new_game(config)
        if result != 0:
            raise RuntimeError("Failed to create new game")

        # Get initial observation
        obs = ffi.new("FcObservation *")
        self._lib.fcgym_get_observation(obs)

        # Update slot mappings
        self._update_slot_mappings(obs)

        # Get valid actions
        valid = ffi.new("FcValidActions *")
        self._lib.fcgym_get_valid_actions(valid)
        self._build_legal_actions(valid)

        # Build observation dict
        observation = self._build_observation(obs)

        info = {
            "turn": obs.turn,
            "action_mask": self._action_mask.copy(),
            "legal_actions": self._legal_actions.copy(),  # Full fixed-size array
            "num_legal_actions": self._num_legal_actions,
        }

        # Free C memory
        self._lib.fcgym_free_observation(obs)
        self._lib.fcgym_free_valid_actions(valid)

        return observation, info

    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Execute one step in the environment."""
        # Decode action
        action_type, actor_id, target_id, sub_target = self._decode_action(action)

        # Create FcAction
        fc_action = ffi.new("FcAction *")
        fc_action.type = action_type
        fc_action.actor_id = actor_id
        fc_action.target_id = target_id
        fc_action.sub_target = sub_target

        # Execute action
        result = self._lib.fcgym_step(fc_action)

        # Get new observation
        obs = ffi.new("FcObservation *")
        self._lib.fcgym_get_observation(obs)

        # Update slot mappings
        self._update_slot_mappings(obs)

        # Get valid actions
        valid = ffi.new("FcValidActions *")
        self._lib.fcgym_get_valid_actions(valid)
        self._build_legal_actions(valid)

        # Build observation dict
        observation = self._build_observation(obs)

        # Use reward from step result, or calculate our own
        reward = result.reward if result.reward != 0.0 else self._calculate_reward(obs)

        terminated = result.done or obs.game_over
        truncated = result.truncated

        info = {
            "turn": obs.turn,
            "action_mask": self._action_mask.copy(),
            "legal_actions": self._legal_actions.copy(),  # Full fixed-size array
            "num_legal_actions": self._num_legal_actions,
            "step_info": ffi.string(result.info).decode('utf-8') if result.info != ffi.NULL else "",
        }

        # Free C memory
        self._lib.fcgym_free_observation(obs)
        self._lib.fcgym_free_valid_actions(valid)

        return observation, reward, terminated, truncated, info

    def _calculate_reward(self, obs) -> float:
        """Calculate reward from observation."""
        # Simple reward based on game state
        controlled = obs.controlled_player

        # Find our player
        our_player = None
        for i in range(obs.num_players):
            if obs.players[i].index == controlled:
                our_player = obs.players[i]
                break

        if our_player is None or not our_player.is_alive:
            return -1.0  # We lost

        # Positive reward for staying alive, units, cities
        reward = 0.001  # Survival reward
        reward += 0.001 * our_player.num_units
        reward += 0.01 * our_player.num_cities
        reward += 0.0001 * our_player.score

        if obs.game_over:
            # Check if we won
            # Simple heuristic: highest score
            max_score = our_player.score
            for i in range(obs.num_players):
                p = obs.players[i]
                if p.is_alive and p.index != controlled and p.score > max_score:
                    max_score = p.score

            if our_player.score >= max_score:
                reward += 1.0  # Win bonus
            else:
                reward -= 0.5  # Loss penalty

        return reward

    def render(self) -> Optional[str]:
        """Render the current game state."""
        if self.render_mode == "human":
            print(f"Turn: {self._num_legal_actions} legal actions")
            return None
        elif self.render_mode == "ansi":
            return f"Legal actions: {self._num_legal_actions}"
        return None

    def close(self):
        """Clean up instance resources.

        Note: This does NOT shutdown the fcgym library. The library remains
        initialized for other env instances and is automatically shutdown
        on process exit via atexit. To explicitly shutdown, use shutdown_library().
        """
        # Clear instance state but don't touch the shared library
        self._initialized = False
        self._lib = None
        self._unit_id_to_slot.clear()
        self._city_id_to_slot.clear()
        self._slot_to_unit_id.clear()
        self._slot_to_city_id.clear()


def make_freeciv_gym_env(**kwargs) -> FreecivGymEnv:
    """Create a FreecivGymEnv with default settings."""
    return FreecivGymEnv(**kwargs)
