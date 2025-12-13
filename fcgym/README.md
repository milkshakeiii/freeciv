# fcgym - Freeciv Gymnasium Environment Wrapper

Synchronous C wrapper around Freeciv's game engine for reinforcement learning.

## Overview

fcgym provides direct C API access to Freeciv without the network layer:

- **Synchronous execution**: No network delays
- **Direct state access**: Read game state from memory
- **Turn-based**: Players alternate (no simultaneous turns)
- **RL-friendly**: Observation/action/step interface

## Building

### 1. Build Freeciv first

```bash
cd freeciv
meson setup build
cd build
meson compile
```

### 2. Build fcgym

```bash
cd fcgym
make BUILD_DIR=../build
make test
```

## Python Gymnasium Env

`freeciv_gym_env.py` exposes a Gymnasium `Env` (`FreecivGymEnv`) backed by `libfcgym.so`.

Demo:

```bash
python3 freeciv/fcgym/demo_freeciv_gym_env.py --steps 50 --seed 123
```

Notes:
- `fcgym` uses global Freeciv state: run at most 1 env instance per process (use subprocess/Ray for parallelism).

## API

```c
#include "fcgym.h"

// Initialize
fcgym_init();

// New game
FcGameConfig config = {
    .ruleset = "civ2civ3",
    .map_xsize = 40,
    .map_ysize = 40,
    .num_ai_players = 3,
    .ai_skill_level = 3,
    .seed = 12345,
    .fog_of_war = true,
};
fcgym_new_game(&config);

// Game loop
FcObservation obs = {0};
while (!obs.game_over) {
    fcgym_get_observation(&obs);

    FcAction action = {
        .type = FCGYM_ACTION_END_TURN,
    };
    FcStepResult result = fcgym_step(&action);
}

// Cleanup
fcgym_free_observation(&obs);
fcgym_shutdown();
```

## Action Types

- `FCGYM_ACTION_UNIT_MOVE` - Move unit (direction in sub_target)
- `FCGYM_ACTION_UNIT_FORTIFY` - Fortify unit
- `FCGYM_ACTION_UNIT_BUILD_CITY` - Found city
- `FCGYM_ACTION_CITY_BUILD` - Change production
- `FCGYM_ACTION_END_TURN` - End turn
- `FCGYM_ACTION_NOOP` - Do nothing

## Status

Work in progress. Core structure is in place, needs testing and completion.
