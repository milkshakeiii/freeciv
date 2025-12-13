/*
 * fcgym.h - Freeciv Gymnasium Environment Wrapper
 *
 * Synchronous C API for using Freeciv as an RL environment.
 * Bypasses the network layer to allow direct game state access and control.
 */

#ifndef FCGYM_H
#define FCGYM_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdbool.h>
#include <stdint.h>

/* Forward declarations from freeciv */
struct player;
struct unit;
struct city;
struct tile;

/*
 * Game configuration for new games
 */
typedef struct {
    const char *ruleset;        /* e.g., "civ2civ3", "classic" */
    int map_xsize;              /* Map width */
    int map_ysize;              /* Map height */
    int num_ai_players;         /* Number of AI opponents */
    int ai_skill_level;         /* 0-10 AI difficulty */
    unsigned int seed;          /* Random seed (0 for random) */
    bool fog_of_war;            /* Enable fog of war */
} FcGameConfig;

/*
 * Action types the agent can take
 */
typedef enum {
    FCGYM_ACTION_UNIT_MOVE,         /* Move unit in direction */
    FCGYM_ACTION_UNIT_ATTACK,       /* Attack target */
    FCGYM_ACTION_UNIT_FORTIFY,      /* Fortify unit */
    FCGYM_ACTION_UNIT_BUILD_CITY,   /* Found city */
    FCGYM_ACTION_UNIT_BUILD_ROAD,   /* Build road/railroad */
    FCGYM_ACTION_UNIT_BUILD_IRRIGATION, /* Build irrigation */
    FCGYM_ACTION_UNIT_BUILD_MINE,   /* Build mine */
    FCGYM_ACTION_UNIT_DISBAND,      /* Disband unit */
    FCGYM_ACTION_CITY_BUILD,        /* Change city production */
    FCGYM_ACTION_CITY_BUY,          /* Buy current production */
    FCGYM_ACTION_RESEARCH_SET,      /* Set research target */
    FCGYM_ACTION_END_TURN,          /* End turn for player */
    FCGYM_ACTION_NOOP,              /* Do nothing */
    FCGYM_ACTION_COUNT
} FcActionType;

/*
 * Action structure for agent commands
 */
typedef struct {
    FcActionType type;
    int actor_id;           /* Unit ID or City ID */
    int target_id;          /* Target tile index, unit ID, or building/unit type */
    int sub_target;         /* Secondary target (e.g., direction for move) */
} FcAction;

/*
 * Tile observation data
 */
typedef struct {
    int terrain;            /* Terrain type index */
    int owner;              /* Owner player index (-1 if none) */
    bool has_city;          /* City present */
    bool has_unit;          /* Unit(s) present */
    bool visible;           /* Visible to agent's player */
    bool explored;          /* Has been explored */
    int8_t extras;          /* Bitmask of extras (roads, irrigation, etc.) */
} FcTileObs;

/*
 * Unit observation data
 */
typedef struct {
    int id;
    int type;               /* Unit type index */
    int owner;              /* Owner player index */
    int tile_index;         /* Location */
    int hp;                 /* Current hit points */
    int max_hp;             /* Maximum hit points */
    int moves_left;         /* Movement points remaining (in fractions) */
    int veteran_level;      /* Veteran status */
    bool fortified;         /* Is fortified */
} FcUnitObs;

/*
 * City observation data
 */
typedef struct {
    int id;
    int owner;              /* Owner player index */
    int tile_index;         /* Location */
    int size;               /* Population size */
    int food_stock;         /* Food in granary */
    int shield_stock;       /* Shields towards production */
    int producing_type;     /* What is being built (-1 if nothing) */
    bool producing_is_unit; /* True if producing unit, false if building */
    int turns_to_complete;  /* Estimated turns to finish production */
} FcCityObs;

/*
 * Player observation data
 */
typedef struct {
    int index;
    bool is_alive;
    bool is_ai;
    int gold;
    int tax_rate;
    int science_rate;
    int luxury_rate;
    int researching;        /* Tech being researched (-1 if none) */
    int research_bulbs;     /* Bulbs accumulated */
    int num_cities;
    int num_units;
    int score;
} FcPlayerObs;

/*
 * Full game observation
 */
typedef struct {
    /* Map dimensions */
    int map_xsize;
    int map_ysize;

    /* Current game state */
    int turn;
    int year;
    int phase;
    int current_player;     /* Index of player whose turn it is */
    int controlled_player;  /* Index of player we control */

    /* Tile data (map_xsize * map_ysize elements) */
    FcTileObs *tiles;
    int num_tiles;

    /* Units visible to controlled player */
    FcUnitObs *units;
    int num_units;
    int max_units;          /* Allocated size */

    /* Cities visible to controlled player */
    FcCityObs *cities;
    int num_cities;
    int max_cities;         /* Allocated size */

    /* Player info */
    FcPlayerObs *players;
    int num_players;

    /* Game over flags */
    bool game_over;
    int winner;             /* Player index of winner (-1 if no winner yet) */
} FcObservation;

/*
 * Valid action mask
 */
typedef struct {
    /* For each unit owned by controlled player */
    struct {
        int unit_id;
        bool can_move[8];       /* 8 directions (non-combat moves only) */
        int attackable_tiles[8]; /* Tile indices of valid attack targets */
        int num_attackable_tiles; /* Number of valid attack targets (0-8) */
        bool can_fortify;
        bool can_build_city;
        bool can_build_road;
        bool can_build_irrigation;
        bool can_build_mine;
        bool can_disband;
    } *unit_actions;
    int num_unit_actions;

    /* For each city owned by controlled player */
    struct {
        int city_id;
        int *buildable_units;   /* Unit type indices that can be built */
        int num_buildable_units;
        int *buildable_buildings; /* Building type indices */
        int num_buildable_buildings;
        bool can_buy;
    } *city_actions;
    int num_city_actions;

    /* Research options */
    int *researchable_techs;
    int num_researchable_techs;

    /* Can end turn */
    bool can_end_turn;
} FcValidActions;

/*
 * Step result
 */
typedef struct {
    float reward;           /* Reward signal */
    bool done;              /* Episode terminated */
    bool truncated;         /* Episode truncated (e.g., max turns) */
    const char *info;       /* Additional info string */
} FcStepResult;


/* ============== API Functions ============== */

/*
 * Initialize the fcgym library. Must be called once before any other functions.
 * Returns 0 on success, non-zero on failure.
 */
int fcgym_init(void);

/*
 * Cleanup and shutdown. Call when done.
 */
void fcgym_shutdown(void);

/*
 * Start a new game with the given configuration.
 * Returns 0 on success, non-zero on failure.
 */
int fcgym_new_game(const FcGameConfig *config);

/*
 * Reset the current game to initial state (faster than new_game).
 * Returns 0 on success, non-zero on failure.
 */
int fcgym_reset(void);

/*
 * Get the current observation. Caller must provide allocated FcObservation.
 * Internal arrays (tiles, units, cities, players) will be allocated/reallocated as needed.
 */
void fcgym_get_observation(FcObservation *obs);

/*
 * Free the internal arrays of an FcObservation.
 */
void fcgym_free_observation(FcObservation *obs);

/*
 * Get valid actions for the controlled player.
 * Caller must provide allocated FcValidActions.
 */
void fcgym_get_valid_actions(FcValidActions *actions);

/*
 * Free the internal arrays of an FcValidActions.
 */
void fcgym_free_valid_actions(FcValidActions *actions);

/*
 * Execute an action and return the result.
 * The game advances after the action (AI players take their turns if applicable).
 */
FcStepResult fcgym_step(const FcAction *action);

/*
 * Get the number of possible unit types (for action space sizing).
 */
int fcgym_num_unit_types(void);

/*
 * Get the number of possible building types.
 */
int fcgym_num_building_types(void);

/*
 * Get the number of possible technologies.
 */
int fcgym_num_techs(void);

/*
 * Get unit type name by index.
 */
const char* fcgym_unit_type_name(int index);

/*
 * Get building type name by index.
 */
const char* fcgym_building_type_name(int index);

/*
 * Get technology name by index.
 */
const char* fcgym_tech_name(int index);


/* ============== Low-level Access (for debugging/advanced use) ============== */

/*
 * Get direct access to the controlled player struct.
 * Returns NULL if no game is running.
 */
struct player* fcgym_get_controlled_player(void);

/*
 * Get a unit by ID.
 */
struct unit* fcgym_get_unit(int unit_id);

/*
 * Get a city by ID.
 */
struct city* fcgym_get_city(int city_id);

/*
 * Get tile by map coordinates.
 */
struct tile* fcgym_get_tile(int x, int y);


#ifdef __cplusplus
}
#endif

#endif /* FCGYM_H */
