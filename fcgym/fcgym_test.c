/*
 * fcgym_test.c - Simple test for the fcgym wrapper
 */

#include <stdio.h>
#include <stdlib.h>
#include "fcgym.h"

int main(int argc, char **argv)
{
    printf("=== fcgym Test ===\n\n");

    /* Initialize */
    printf("Initializing fcgym...\n");
    if (fcgym_init() != 0) {
        fprintf(stderr, "Failed to initialize fcgym\n");
        return 1;
    }
    printf("Initialized successfully!\n\n");

    /* Create a new game */
    printf("Creating new game...\n");
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

    /* Get initial observation */
    printf("Getting observation...\n");
    FcObservation obs = {0};
    fcgym_get_observation(&obs);

    printf("Map size: %d x %d\n", obs.map_xsize, obs.map_ysize);
    printf("Turn: %d, Year: %d\n", obs.turn, obs.year);
    printf("Number of players: %d\n", obs.num_players);
    printf("Visible units: %d\n", obs.num_units);
    printf("Visible cities: %d\n", obs.num_cities);

    /* Print player info */
    printf("\nPlayers:\n");
    for (int i = 0; i < obs.num_players; i++) {
        FcPlayerObs *p = &obs.players[i];
        printf("  Player %d: %s, Gold: %d, Cities: %d, Units: %d\n",
               p->index, p->is_ai ? "AI" : "Human",
               p->gold, p->num_cities, p->num_units);
    }

    /* Print unit type counts */
    printf("\nRuleset info:\n");
    printf("  Unit types: %d\n", fcgym_num_unit_types());
    printf("  Building types: %d\n", fcgym_num_building_types());
    printf("  Technologies: %d\n", fcgym_num_techs());

    /* Print first few unit types */
    printf("\nSample unit types:\n");
    for (int i = 0; i < 5 && i < fcgym_num_unit_types(); i++) {
        const char *name = fcgym_unit_type_name(i);
        if (name) {
            printf("  %d: %s\n", i, name);
        }
    }

    /* Run a few turns with NOOP actions */
    printf("\nRunning 5 turns with END_TURN actions...\n");
    for (int turn = 0; turn < 5; turn++) {
        FcAction action = {
            .type = FCGYM_ACTION_END_TURN,
            .actor_id = 0,
            .target_id = 0,
            .sub_target = 0,
        };

        FcStepResult result = fcgym_step(&action);
        printf("  Turn %d: done=%d, reward=%.2f\n",
               turn + 1, result.done, result.reward);

        if (result.done) {
            printf("  Game over!\n");
            break;
        }

        /* Update observation */
        fcgym_get_observation(&obs);
        printf("    Now turn %d, player cities: %d, units: %d\n",
               obs.turn, obs.players[0].num_cities, obs.players[0].num_units);
    }

    /* Cleanup */
    printf("\nCleaning up...\n");
    fcgym_free_observation(&obs);
    fcgym_shutdown();

    printf("Test complete!\n");
    return 0;
}
