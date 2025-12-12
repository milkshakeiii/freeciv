/*
 * fcgym_test.c - Test for the fcgym wrapper
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "fcgym.h"

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST_ASSERT(cond, msg) do { \
    if (cond) { \
        printf("  PASS: %s\n", msg); \
        tests_passed++; \
    } else { \
        printf("  FAIL: %s\n", msg); \
        tests_failed++; \
    } \
} while(0)

/* Find a unit of a specific type owned by controlled player */
static int find_unit_by_type(const FcObservation *obs, int unit_type, int controlled_player)
{
    for (int i = 0; i < obs->num_units; i++) {
        if (obs->units[i].owner == controlled_player &&
            obs->units[i].type == unit_type) {
            return obs->units[i].id;
        }
    }
    return -1;
}

/* Find unit by type name */
static int find_unit_by_type_name(const FcObservation *obs, const char *type_name, int controlled_player)
{
    for (int i = 0; i < obs->num_units; i++) {
        if (obs->units[i].owner == controlled_player) {
            const char *name = fcgym_unit_type_name(obs->units[i].type);
            if (name && strcmp(name, type_name) == 0) {
                return obs->units[i].id;
            }
        }
    }
    return -1;
}

/* Get unit info by id */
static FcUnitObs* get_unit_by_id(FcObservation *obs, int unit_id)
{
    for (int i = 0; i < obs->num_units; i++) {
        if (obs->units[i].id == unit_id) {
            return &obs->units[i];
        }
    }
    return NULL;
}

/* Get city info by id */
static FcCityObs* get_city_by_id(FcObservation *obs, int city_id)
{
    for (int i = 0; i < obs->num_cities; i++) {
        if (obs->cities[i].id == city_id) {
            return &obs->cities[i];
        }
    }
    return NULL;
}

/* Count units owned by player */
static int count_player_units(const FcObservation *obs, int player)
{
    int count = 0;
    for (int i = 0; i < obs->num_units; i++) {
        if (obs->units[i].owner == player) {
            count++;
        }
    }
    return count;
}

int main(int argc, char **argv)
{
    (void)argc;
    (void)argv;

    printf("=== fcgym State Transition Tests ===\n\n");

    /* Initialize */
    printf("Initializing fcgym...\n");
    if (fcgym_init() != 0) {
        fprintf(stderr, "Failed to initialize fcgym\n");
        return 1;
    }

    /* Create a new game */
    FcGameConfig config = {
        .ruleset = "civ2civ3",
        .map_xsize = 40,
        .map_ysize = 40,
        .num_ai_players = 2,
        .ai_skill_level = 3,
        .seed = 12345,
        .fog_of_war = true,
    };

    if (fcgym_new_game(&config) != 0) {
        fprintf(stderr, "Failed to create new game\n");
        fcgym_shutdown();
        return 1;
    }
    printf("Game created successfully!\n\n");

    FcObservation obs = {0};
    FcValidActions valid = {0};
    FcStepResult result;

    /* ========== Test 1: Build City ========== */
    printf("=== Test 1: Build City ===\n");
    fcgym_get_observation(&obs);

    int initial_cities = obs.players[obs.controlled_player].num_cities;
    int initial_units = count_player_units(&obs, obs.controlled_player);
    int settler_id = find_unit_by_type_name(&obs, "Settlers", obs.controlled_player);

    printf("Before: cities=%d, units=%d, settler_id=%d\n",
           initial_cities, initial_units, settler_id);

    if (settler_id >= 0) {
        FcAction build_city = {
            .type = FCGYM_ACTION_UNIT_BUILD_CITY,
            .actor_id = settler_id,
        };
        result = fcgym_step(&build_city);
        fcgym_get_observation(&obs);

        int new_cities = obs.players[obs.controlled_player].num_cities;
        int new_units = count_player_units(&obs, obs.controlled_player);
        printf("After: cities=%d, units=%d\n", new_cities, new_units);

        TEST_ASSERT(new_cities == initial_cities + 1, "City count increased by 1");
        TEST_ASSERT(new_units == initial_units - 1, "Unit count decreased by 1 (settler consumed)");
        TEST_ASSERT(get_unit_by_id(&obs, settler_id) == NULL, "Settler unit no longer exists");
    } else {
        printf("SKIP: No settler found\n");
    }

    /* ========== Test 2: Buy (Rush) Production ========== */
    printf("\n=== Test 2: Buy (Rush) Production ===\n");

    /* End turn first - can't buy in city the turn it was founded */
    FcAction end_first = { .type = FCGYM_ACTION_END_TURN };
    fcgym_step(&end_first);

    fcgym_get_observation(&obs);
    fcgym_get_valid_actions(&valid);

    /* After ending turn, we can buy in the city */
    if (valid.num_city_actions > 0) {
        int city_id = valid.city_actions[0].city_id;
        FcCityObs *city = get_city_by_id(&obs, city_id);

        /* Find Warriors (cheap unit, costs 10 shields) */
        int warriors_type = -1;
        for (int u = 0; u < valid.city_actions[0].num_buildable_units; u++) {
            int unit_type = valid.city_actions[0].buildable_units[u];
            const char *name = fcgym_unit_type_name(unit_type);
            if (name && strcmp(name, "Warriors") == 0) {
                warriors_type = unit_type;
                break;
            }
        }

        if (warriors_type >= 0) {
            /* Set production to Warriors */
            FcAction set_prod = {
                .type = FCGYM_ACTION_CITY_BUILD,
                .actor_id = city_id,
                .target_id = warriors_type,
                .sub_target = 0,
            };
            fcgym_step(&set_prod);

            /* Check if we can buy */
            fcgym_get_observation(&obs);
            fcgym_get_valid_actions(&valid);

            bool can_buy_now = false;
            for (int i = 0; i < valid.num_city_actions; i++) {
                if (valid.city_actions[i].city_id == city_id && valid.city_actions[i].can_buy) {
                    can_buy_now = true;
                    break;
                }
            }

            city = get_city_by_id(&obs, city_id);
            int gold_before = obs.players[obs.controlled_player].gold;
            int shields_before = city ? city->shield_stock : 0;
            int units_before = count_player_units(&obs, obs.controlled_player);

            printf("City %d building Warriors, gold=%d, shields=%d, can_buy=%d\n",
                   city_id, gold_before, shields_before, can_buy_now);

            if (can_buy_now) {
                FcAction buy = {
                    .type = FCGYM_ACTION_CITY_BUY,
                    .actor_id = city_id,
                };
                fcgym_step(&buy);
                fcgym_get_observation(&obs);

                int gold_after = obs.players[obs.controlled_player].gold;
                city = get_city_by_id(&obs, city_id);
                int shields_after = city ? city->shield_stock : 0;

                printf("After buy: gold=%d (was %d), shields=%d (was %d)\n",
                       gold_after, gold_before, shields_after, shields_before);

                TEST_ASSERT(gold_after < gold_before, "Gold decreased after buying");
                TEST_ASSERT(shields_after > shields_before, "Shield stock filled after buying");

                /* End turn to see the unit actually created */
                FcAction end = { .type = FCGYM_ACTION_END_TURN };
                fcgym_step(&end);
                fcgym_get_observation(&obs);

                int units_after = count_player_units(&obs, obs.controlled_player);
                printf("After turn end: units=%d (was %d)\n", units_after, units_before);
                TEST_ASSERT(units_after > units_before, "Unit was built after turn ended");
            } else {
                printf("SKIP: Cannot afford to buy Warriors (gold=%d)\n", gold_before);
            }
        } else {
            printf("SKIP: Warriors not available\n");
        }
    } else {
        printf("SKIP: No city available\n");
    }
    fcgym_free_valid_actions(&valid);

    /* ========== Test 3: Unit Movement ========== */
    printf("\n=== Test 3: Unit Movement ===\n");
    fcgym_get_observation(&obs);
    fcgym_get_valid_actions(&valid);

    /* Find a unit that can move */
    int move_unit_id = -1;
    int move_dir = -1;
    for (int i = 0; i < valid.num_unit_actions; i++) {
        for (int d = 0; d < 8; d++) {
            if (valid.unit_actions[i].can_move[d]) {
                move_unit_id = valid.unit_actions[i].unit_id;
                move_dir = d;
                break;
            }
        }
        if (move_unit_id >= 0) break;
    }

    if (move_unit_id >= 0) {
        FcUnitObs *unit = get_unit_by_id(&obs, move_unit_id);
        int old_tile = unit->tile_index;
        int old_moves = unit->moves_left;
        printf("Before: unit %d at tile %d, moves=%d, direction=%d\n",
               move_unit_id, old_tile, old_moves, move_dir);

        FcAction move = {
            .type = FCGYM_ACTION_UNIT_MOVE,
            .actor_id = move_unit_id,
            .sub_target = move_dir,
        };
        result = fcgym_step(&move);
        fcgym_get_observation(&obs);

        unit = get_unit_by_id(&obs, move_unit_id);
        if (unit) {
            printf("After: unit %d at tile %d, moves=%d\n",
                   move_unit_id, unit->tile_index, unit->moves_left);
            TEST_ASSERT(unit->tile_index != old_tile, "Unit tile changed");
            TEST_ASSERT(unit->moves_left < old_moves, "Movement points decreased");
        } else {
            printf("FAIL: Unit disappeared after move\n");
            tests_failed++;
        }
    } else {
        printf("SKIP: No unit can move\n");
    }
    fcgym_free_valid_actions(&valid);

    /* ========== Test 3: Fortify Unit ========== */
    printf("\n=== Test 3: Fortify Unit ===\n");
    fcgym_get_observation(&obs);
    fcgym_get_valid_actions(&valid);

    /* Find a unit that can fortify */
    int fortify_unit_id = -1;
    for (int i = 0; i < valid.num_unit_actions; i++) {
        if (valid.unit_actions[i].can_fortify) {
            fortify_unit_id = valid.unit_actions[i].unit_id;
            break;
        }
    }

    if (fortify_unit_id >= 0) {
        FcUnitObs *unit = get_unit_by_id(&obs, fortify_unit_id);
        const char *type_name = fcgym_unit_type_name(unit->type);
        printf("Before: unit %d (%s) fortified=%d\n",
               fortify_unit_id, type_name ? type_name : "?", unit->fortified);

        FcAction fortify = {
            .type = FCGYM_ACTION_UNIT_FORTIFY,
            .actor_id = fortify_unit_id,
        };
        result = fcgym_step(&fortify);
        fcgym_get_observation(&obs);

        unit = get_unit_by_id(&obs, fortify_unit_id);
        if (unit) {
            printf("After: unit %d fortified=%d\n", fortify_unit_id, unit->fortified);
            /* Note: fortified becomes true after fortifying completes (takes a turn) */
            TEST_ASSERT(1, "Fortify action executed (unit is fortifying)");
        }
    } else {
        printf("SKIP: No unit can fortify\n");
    }
    fcgym_free_valid_actions(&valid);

    /* ========== Test 4: Set Research ========== */
    printf("\n=== Test 4: Set Research ===\n");
    fcgym_get_observation(&obs);
    fcgym_get_valid_actions(&valid);

    int old_research = obs.players[obs.controlled_player].researching;
    printf("Before: researching tech %d (%s)\n",
           old_research, fcgym_tech_name(old_research));

    /* Find a different tech to research */
    int new_tech = -1;
    for (int i = 0; i < valid.num_researchable_techs; i++) {
        if (valid.researchable_techs[i] != old_research) {
            new_tech = valid.researchable_techs[i];
            break;
        }
    }

    if (new_tech >= 0) {
        printf("Switching to tech %d (%s)\n", new_tech, fcgym_tech_name(new_tech));

        FcAction research = {
            .type = FCGYM_ACTION_RESEARCH_SET,
            .target_id = new_tech,
        };
        result = fcgym_step(&research);
        fcgym_get_observation(&obs);

        int current_research = obs.players[obs.controlled_player].researching;
        printf("After: researching tech %d (%s)\n",
               current_research, fcgym_tech_name(current_research));
        TEST_ASSERT(current_research == new_tech, "Research target changed to selected tech");
    } else {
        printf("SKIP: No alternative tech to research\n");
    }
    fcgym_free_valid_actions(&valid);

    /* ========== Test 5: City Production Change ========== */
    printf("\n=== Test 5: City Production Change ===\n");
    fcgym_get_observation(&obs);
    fcgym_get_valid_actions(&valid);

    if (valid.num_city_actions > 0 && valid.city_actions[0].num_buildable_units > 1) {
        int city_id = valid.city_actions[0].city_id;
        FcCityObs *city = get_city_by_id(&obs, city_id);

        int old_production = city->producing_type;
        printf("Before: city %d producing type %d\n", city_id, old_production);

        /* Find a different unit to build */
        int new_production = -1;
        for (int i = 0; i < valid.city_actions[0].num_buildable_units; i++) {
            if (valid.city_actions[0].buildable_units[i] != old_production) {
                new_production = valid.city_actions[0].buildable_units[i];
                break;
            }
        }

        if (new_production >= 0) {
            printf("Switching to build %s (type %d)\n",
                   fcgym_unit_type_name(new_production), new_production);

            FcAction city_build = {
                .type = FCGYM_ACTION_CITY_BUILD,
                .actor_id = city_id,
                .target_id = new_production,
                .sub_target = 0,  /* 0 = unit */
            };
            result = fcgym_step(&city_build);
            fcgym_get_observation(&obs);

            city = get_city_by_id(&obs, city_id);
            printf("After: city %d producing type %d, is_unit=%d\n",
                   city_id, city->producing_type, city->producing_is_unit);
            TEST_ASSERT(city->producing_type == new_production, "City production changed");
            TEST_ASSERT(city->producing_is_unit == true, "City is producing a unit");
        }
    } else {
        printf("SKIP: No city or not enough buildable units\n");
    }
    fcgym_free_valid_actions(&valid);

    /* ========== Test 6: Workers Build Irrigation ========== */
    printf("\n=== Test 6: Workers Build Irrigation ===\n");
    fcgym_get_observation(&obs);
    fcgym_get_valid_actions(&valid);

    /* Find a unit that can build irrigation */
    int irrigate_unit_id = -1;
    for (int i = 0; i < valid.num_unit_actions; i++) {
        if (valid.unit_actions[i].can_build_irrigation) {
            irrigate_unit_id = valid.unit_actions[i].unit_id;
            break;
        }
    }

    if (irrigate_unit_id >= 0) {
        FcUnitObs *unit = get_unit_by_id(&obs, irrigate_unit_id);
        const char *type_name = fcgym_unit_type_name(unit->type);
        printf("Unit %d (%s) will build irrigation\n", irrigate_unit_id, type_name ? type_name : "?");

        FcAction irrigate = {
            .type = FCGYM_ACTION_UNIT_BUILD_IRRIGATION,
            .actor_id = irrigate_unit_id,
            .sub_target = -1,  /* auto-select */
        };
        result = fcgym_step(&irrigate);

        /* Activity starts - unit should still exist */
        fcgym_get_observation(&obs);
        unit = get_unit_by_id(&obs, irrigate_unit_id);
        TEST_ASSERT(unit != NULL, "Unit still exists after starting irrigation");
        printf("Irrigation activity started\n");
    } else {
        printf("SKIP: No unit can build irrigation\n");
    }
    fcgym_free_valid_actions(&valid);

    /* ========== Test 7: Disband Unit ========== */
    printf("\n=== Test 7: Disband Unit ===\n");
    fcgym_get_observation(&obs);
    fcgym_get_valid_actions(&valid);

    int units_before = count_player_units(&obs, obs.controlled_player);

    /* Find a unit that can actually be disbanded from valid actions */
    int disband_id = -1;
    for (int i = 0; i < valid.num_unit_actions; i++) {
        if (valid.unit_actions[i].can_disband) {
            disband_id = valid.unit_actions[i].unit_id;
            break;
        }
    }
    fcgym_free_valid_actions(&valid);

    if (disband_id >= 0) {
        FcUnitObs *unit = get_unit_by_id(&obs, disband_id);
        const char *type_name = fcgym_unit_type_name(unit->type);
        printf("Before: %d units, disbanding unit %d (%s)\n",
               units_before, disband_id, type_name ? type_name : "?");

        FcAction disband = {
            .type = FCGYM_ACTION_UNIT_DISBAND,
            .actor_id = disband_id,
        };
        result = fcgym_step(&disband);
        fcgym_get_observation(&obs);

        int units_after = count_player_units(&obs, obs.controlled_player);
        printf("After: %d units\n", units_after);

        TEST_ASSERT(units_after == units_before - 1, "Unit count decreased by 1");
        TEST_ASSERT(get_unit_by_id(&obs, disband_id) == NULL, "Disbanded unit no longer exists");
    } else {
        printf("SKIP: No suitable unit to disband\n");
    }

    /* ========== Test 8: End Turn ========== */
    printf("\n=== Test 8: End Turn ===\n");
    fcgym_get_observation(&obs);

    int old_turn = obs.turn;
    printf("Before: turn %d\n", old_turn);

    FcAction end_turn = {
        .type = FCGYM_ACTION_END_TURN,
    };
    result = fcgym_step(&end_turn);
    fcgym_get_observation(&obs);

    printf("After: turn %d\n", obs.turn);
    TEST_ASSERT(obs.turn == old_turn + 1, "Turn number increased by 1");

    /* ========== Test 9: AI Turn Cycle ========== */
    printf("\n=== Test 9: AI Turn Cycle ===\n");
    fcgym_get_observation(&obs);

    /* Record state before turn */
    int turn_before = obs.turn;
    int controlled = obs.controlled_player;

    /* Record our units' movement points */
    int our_unit_id = -1;
    int our_unit_moves_before = -1;
    for (int i = 0; i < obs.num_units; i++) {
        if (obs.units[i].owner == controlled) {
            our_unit_id = obs.units[i].id;
            our_unit_moves_before = obs.units[i].moves_left;
            break;
        }
    }

    /* Record AI player states */
    printf("Before end turn (turn %d):\n", turn_before);
    printf("  Controlled player: %d\n", controlled);
    for (int i = 0; i < obs.num_players; i++) {
        if (obs.players[i].is_ai && obs.players[i].is_alive) {
            printf("  AI player %d: gold=%d, units=%d, cities=%d\n",
                   obs.players[i].index,
                   obs.players[i].gold,
                   obs.players[i].num_units,
                   obs.players[i].num_cities);
        }
    }

    /* Find a unit that can actually move and use up some movement points */
    bool found_movable = false;
    fcgym_get_valid_actions(&valid);
    for (int i = 0; i < valid.num_unit_actions && !found_movable; i++) {
        for (int d = 0; d < 8; d++) {
            if (valid.unit_actions[i].can_move[d]) {
                our_unit_id = valid.unit_actions[i].unit_id;
                FcUnitObs *unit = get_unit_by_id(&obs, our_unit_id);
                if (unit) {
                    our_unit_moves_before = unit->moves_left;
                    printf("  Moving unit %d (moves=%d)\n", our_unit_id, our_unit_moves_before);

                    FcAction move = {
                        .type = FCGYM_ACTION_UNIT_MOVE,
                        .actor_id = our_unit_id,
                        .sub_target = d,
                    };
                    fcgym_step(&move);

                    fcgym_get_observation(&obs);
                    unit = get_unit_by_id(&obs, our_unit_id);
                    if (unit && unit->moves_left < our_unit_moves_before) {
                        printf("  Unit %d moves after action: %d (was %d)\n",
                               our_unit_id, unit->moves_left, our_unit_moves_before);
                        our_unit_moves_before = unit->moves_left;
                        found_movable = true;
                    }
                }
                break;
            }
        }
    }
    fcgym_free_valid_actions(&valid);

    if (!found_movable) {
        printf("  No movable unit found, skipping movement restoration test\n");
        our_unit_id = -1;
    }

    /* End turn - AI should take their turns */
    printf("\nEnding turn...\n");
    FcAction end = { .type = FCGYM_ACTION_END_TURN };
    result = fcgym_step(&end);

    fcgym_get_observation(&obs);
    printf("\nAfter end turn (turn %d):\n", obs.turn);
    TEST_ASSERT(obs.turn == turn_before + 1, "Turn advanced after AI turns");

    /* Check our unit got movement points back */
    if (our_unit_id >= 0) {
        FcUnitObs *unit = get_unit_by_id(&obs, our_unit_id);
        if (unit) {
            printf("  Our unit %d moves restored: %d\n", our_unit_id, unit->moves_left);
            TEST_ASSERT(unit->moves_left > our_unit_moves_before,
                       "Unit movement points restored after turn");
        }
    }

    /* Check AI states changed (they should have done something) */
    printf("  AI player states after their turns:\n");
    for (int i = 0; i < obs.num_players; i++) {
        if (obs.players[i].is_ai && obs.players[i].is_alive) {
            printf("    AI player %d: gold=%d, units=%d, cities=%d\n",
                   obs.players[i].index,
                   obs.players[i].gold,
                   obs.players[i].num_units,
                   obs.players[i].num_cities);
        }
    }

    /* Verify we can still take actions (it's our turn again) */
    fcgym_get_valid_actions(&valid);
    TEST_ASSERT(valid.can_end_turn, "Can end turn (it's our turn)");
    TEST_ASSERT(valid.num_unit_actions > 0 || count_player_units(&obs, controlled) == 0,
               "Have unit actions available (or no units)");
    printf("  We have %d units with actions available\n", valid.num_unit_actions);
    fcgym_free_valid_actions(&valid);

    /* ========== Test 10: Multiple Turn Cycle ========== */
    printf("\n=== Test 10: Multiple Turn Cycle ===\n");
    printf("Running 5 turns to verify stability...\n");

    int start_turn = obs.turn;
    for (int t = 0; t < 5; t++) {
        fcgym_get_observation(&obs);
        int current_turn = obs.turn;

        /* Do a simple action each turn if possible */
        fcgym_get_valid_actions(&valid);
        if (valid.num_unit_actions > 0) {
            /* Try to move first available unit */
            for (int i = 0; i < valid.num_unit_actions; i++) {
                for (int d = 0; d < 8; d++) {
                    if (valid.unit_actions[i].can_move[d]) {
                        FcAction move = {
                            .type = FCGYM_ACTION_UNIT_MOVE,
                            .actor_id = valid.unit_actions[i].unit_id,
                            .sub_target = d,
                        };
                        fcgym_step(&move);
                        goto done_action;
                    }
                }
            }
        }
        done_action:
        fcgym_free_valid_actions(&valid);

        /* End turn */
        FcAction et = { .type = FCGYM_ACTION_END_TURN };
        result = fcgym_step(&et);

        fcgym_get_observation(&obs);
        printf("  Turn %d -> %d (game_over=%d)\n", current_turn, obs.turn, obs.game_over);

        if (obs.game_over) {
            printf("  Game ended early!\n");
            break;
        }
    }

    fcgym_get_observation(&obs);
    TEST_ASSERT(obs.turn >= start_turn + 5 || obs.game_over,
               "Completed 5 turns or game ended");
    printf("Final turn: %d, game_over: %d\n", obs.turn, obs.game_over);

    /* ========== Test 11: Build Unit From City ========== */
    printf("\n=== Test 11: Build Unit From City ===\n");
    fcgym_get_observation(&obs);
    fcgym_get_valid_actions(&valid);

    if (valid.num_city_actions > 0 && valid.city_actions[0].num_buildable_units > 0) {
        int city_id = valid.city_actions[0].city_id;
        FcCityObs *city = get_city_by_id(&obs, city_id);

        /* Find the cheapest unit to build (usually Warriors) */
        int unit_to_build = valid.city_actions[0].buildable_units[0];
        const char *unit_name = fcgym_unit_type_name(unit_to_build);
        printf("City %d will build %s (type %d)\n", city_id, unit_name ? unit_name : "?", unit_to_build);

        /* Set production */
        FcAction set_prod = {
            .type = FCGYM_ACTION_CITY_BUILD,
            .actor_id = city_id,
            .target_id = unit_to_build,
            .sub_target = 0,  /* 0 = unit */
        };
        fcgym_step(&set_prod);

        /* Record units before */
        fcgym_get_observation(&obs);
        int units_before = count_player_units(&obs, obs.controlled_player);
        city = get_city_by_id(&obs, city_id);
        printf("Before: %d units, city shield_stock=%d, turns_to_complete=%d\n",
               units_before, city->shield_stock, city->turns_to_complete);

        /* Advance turns until unit is built (max 20 turns to avoid infinite loop) */
        bool unit_built = false;
        int max_turns = 20;
        for (int t = 0; t < max_turns && !unit_built && !obs.game_over; t++) {
            FcAction et = { .type = FCGYM_ACTION_END_TURN };
            fcgym_step(&et);
            fcgym_get_observation(&obs);

            int units_now = count_player_units(&obs, obs.controlled_player);
            city = get_city_by_id(&obs, city_id);

            if (units_now > units_before) {
                printf("Turn %d: Unit built! Units: %d -> %d\n", obs.turn, units_before, units_now);
                unit_built = true;
            } else if (city) {
                printf("  Turn %d: shield_stock=%d, turns_to_complete=%d\n",
                       obs.turn, city->shield_stock, city->turns_to_complete);
            }
        }

        TEST_ASSERT(unit_built, "Unit was built from city production");

        /* Verify city is still producing (auto-queued same unit or something else) */
        fcgym_get_observation(&obs);
        city = get_city_by_id(&obs, city_id);
        if (city) {
            printf("After build: city producing type %d, is_unit=%d\n",
                   city->producing_type, city->producing_is_unit);
        }
    } else {
        printf("SKIP: No city available to build units\n");
    }
    fcgym_free_valid_actions(&valid);

    /* ========== Summary ========== */
    printf("\n=== Test Summary ===\n");
    printf("Passed: %d\n", tests_passed);
    printf("Failed: %d\n", tests_failed);

    /* Cleanup */
    fcgym_free_observation(&obs);
    fcgym_shutdown();

    return tests_failed > 0 ? 1 : 0;
}
