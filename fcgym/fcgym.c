/*
 * fcgym.c - Freeciv Gymnasium Environment Wrapper Implementation
 *
 * This file implements the synchronous wrapper around Freeciv's game engine.
 */

#ifdef HAVE_CONFIG_H
#include <fc_config.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Freeciv utility headers */
#include "fc_prehdrs.h"
#include "genlist.h"
#include "log.h"
#include "mem.h"
#include "rand.h"
#include "registry.h"
#include "support.h"

/* Freeciv common headers */
#include "actions.h"
#include "ai.h"
#include "city.h"
#include "fc_types.h"
#include "game.h"
#include "improvement.h"
#include "map.h"
#include "movement.h"
#include "nation.h"
#include "packets.h"
#include "player.h"
#include "research.h"
#include "tech.h"
#include "tile.h"
#include "unit.h"
#include "unitlist.h"
#include "unittype.h"
#include "world_object.h"

/* Freeciv server headers */
#include "aiiface.h"
#include "animals.h"
#include "citytools.h"
#include "cityhand.h"
#include "cityturn.h"
#include "gamehand.h"
#include "mapgen.h"
#include "maphand.h"
#include "plrhand.h"
#include "ruleload.h"
#include "sanitycheck.h"
#include "sernet.h"
#include "settings.h"
#include "srv_main.h"
#include "stdinhand.h"
#include "edithand.h"
#include "voting.h"
#include "techtools.h"
#include "unithand.h"
#include "unittools.h"

/* Server advisors */
#include "advdata.h"

/* AI headers */
#include "aitraits.h"

/* Server generator */
#include "startpos.h"

/* Server other */
#include "score.h"

/* Our header */
#include "fcgym.h"

/* ============== Internal State ============== */

static bool fcgym_initialized = false;
static bool fcgym_game_running = false;
static int controlled_player_idx = 0;  /* Index of the human-controlled player */

/* ============== Helper Functions ============== */

/*
 * Initialize freeciv internals without starting network.
 */
static int fcgym_init_freeciv(void)
{
    /* Initialize server components */
    srv_init();

    /* Initialize random number generator early (needed by ruleset loading) */
    init_game_seed();

    /* Initialize connection list (needed even without network) */
    init_connections();

    /* Initialize settings (required before loading rulesets) */
    settings_init(TRUE);

    /* Initialize stdin command handler */
    stdinhand_init();

    /* Initialize edit/voting handlers */
    edithand_init();
    voting_init();

    /* Initialize AI timer */
    ai_timer_init();

    /* Initialize game structures */
    server_game_init(FALSE);

    return 0;
}

/*
 * Load a ruleset.
 */
static int fcgym_load_ruleset(const char *ruleset)
{
    if (ruleset == NULL) {
        ruleset = "civ2civ3";  /* Default ruleset */
    }

    sz_strlcpy(game.server.rulesetdir, ruleset);

    if (!load_rulesets(NULL, NULL, FALSE, NULL, TRUE, FALSE, TRUE)) {
        log_error("Failed to load ruleset: %s", ruleset);
        return -1;
    }

    return 0;
}

/*
 * Create players for the game.
 */
static int fcgym_create_players(int num_ai_players, int ai_skill_level)
{
    struct player *pplayer;

    /* Create the human-controlled player */
    pplayer = server_create_player(-1, default_ai_type_name(),
                                   NULL, FALSE);
    if (pplayer == NULL) {
        log_error("Failed to create controlled player");
        return -1;
    }

    controlled_player_idx = player_number(pplayer);
    set_as_human(pplayer);
    server_player_init(pplayer, FALSE, TRUE);

    /* Assign a nation to the human player */
    player_set_nation(pplayer, pick_a_nation(NULL, FALSE, TRUE,
                                              NOT_A_BARBARIAN));
    ai_traits_init(pplayer);  /* Initialize traits based on nation */

    /* Create AI players */
    for (int i = 0; i < num_ai_players; i++) {
        pplayer = server_create_player(-1, default_ai_type_name(),
                                       NULL, FALSE);
        if (pplayer == NULL) {
            log_error("Failed to create AI player %d", i);
            return -1;
        }
        set_as_ai(pplayer);
        pplayer->ai_common.skill_level = ai_skill_level;
        server_player_init(pplayer, FALSE, TRUE);

        /* Assign a nation */
        player_set_nation(pplayer, pick_a_nation(NULL, FALSE, TRUE,
                                                  NOT_A_BARBARIAN));
        ai_traits_init(pplayer);  /* Initialize traits based on nation */
    }

    return 0;
}

/*
 * Generate the map.
 */
static int fcgym_generate_map(int xsize, int ysize, unsigned int seed)
{
    struct unit_type *initial_unit = NULL;
    int i;

    /* Set map size */
    wld.map.xsize = xsize;
    wld.map.ysize = ysize;

    /* Set seed */
    if (seed != 0) {
        game.server.seed = seed;
    }
    init_game_seed();

    /* Get the initial unit type for start position generation */
    int sucount = strlen(game.server.start_units);
    if (sucount > 0) {
        for (i = 0; initial_unit == NULL && i < sucount; i++) {
            initial_unit = crole_to_unit_type(game.server.start_units[i], NULL);
        }
    }
    if (initial_unit == NULL) {
        /* Fallback: first unit the initial city might build */
        initial_unit = get_role_unit(L_FIRSTBUILD, 0);
    }

    /* Allocate map first so we can init player maps */
    main_map_allocate();

    /* Initialize player map data BEFORE map generation */
    /* (normally done after, but start pos generation needs it) */
    players_iterate(pplayer) {
        player_map_init(pplayer);
    } players_iterate_end;

    /* Generate the map using freeciv's map generator */
    /* Pass autosize=FALSE since we already allocated the map */
    if (!map_fractal_generate(FALSE, initial_unit)) {
        log_error("Failed to generate map");
        return -1;
    }

    /* Initialize remaining map data */
    game_map_init();

    return 0;
}

/*
 * Start the game (called after setup is complete).
 * This mirrors the initialization done in srv_ready() in srv_main.c
 */
static int fcgym_start_game(void)
{
    /* Set game as new game - important for proper initialization */
    game.info.is_new_game = TRUE;

    /* CRITICAL: Shuffle players before init_new_game!
     * Without this, shuffled_order is all zeros, and every
     * shuffled_players_iterate returns player 0 */
    shuffle_players();

    /* Pregame turn 0 -> game turn 1 (like srv_ready) */
    game.info.turn++;
    game.info.year = game.server.start_year;

    /* Notify AI that map is ready */
    CALL_FUNC_EACH_AI(map_ready);

    /* Enter running state */
    set_server_state(S_S_RUNNING);

    /* Set fog of war old value (before player map allocation) */
    game.server.fogofwar_old = game.info.fogofwar;

    /* Initialize per-player data that srv_ready does BEFORE init_new_game */
    players_iterate(pplayer) {
        /* player_map_init already called in fcgym_generate_map */

        /* Limit tax/science/luxury rates to valid ranges */
        player_limit_to_max_rates(pplayer);

        /* Set AI difficulty level (this also sets science_cost) */
        if (is_ai(pplayer)) {
            set_ai_level_direct(pplayer, pplayer->ai_common.skill_level);
        } else {
            /* Human players: set science_cost to 100 (normal rate) */
            pplayer->ai_common.science_cost = 100;
        }

        /* Set initial gold and infra points */
        pplayer->economic.gold = game.info.gold;
        pplayer->economic.infra_points = game.info.infrapoints;
    } players_iterate_end;

    /* Initialize technologies - give starting techs as per ruleset */
    researches_iterate(presearch) {
        init_tech(presearch, TRUE);
        give_initial_techs(presearch, game.info.tech);
    } researches_iterate_end;

    /* Assign player colors from ruleset */
    assign_player_colors();

    /* Analyze rulesets for AI advisor */
    players_iterate(pplayer) {
        adv_data_analyze_rulesets(pplayer);
    } players_iterate_end;

    /* Set AI advisor data defaults for new game */
    players_iterate(pplayer) {
        adv_data_default(pplayer);
    } players_iterate_end;

    /* Now call init_new_game to place units and cities */
    init_new_game();

    /* Create animal units on the map */
    create_animals();

    /* Notify AI modules that game has started */
    CALL_FUNC_EACH_AI(game_start);

    /* Start the first turn and phase - matches normal server flow */
    begin_turn(TRUE);
    begin_phase(TRUE);

    return 0;
}

/*
 * Run a single phase for all AI players.
 * This is what would normally happen during server_sniff_all_input().
 */
static void fcgym_run_ai_phase(void)
{
    /* Let AI players take their turns */
    players_iterate(pplayer) {
        if (is_ai(pplayer) && pplayer->is_alive
            && is_player_phase(pplayer, game.info.phase)) {
            CALL_PLR_AI_FUNC(phase_finished, pplayer, pplayer);
            pplayer->ai_phase_done = TRUE;
        }
    } players_iterate_end;
}

/*
 * Process end of phase for all players.
 */
static void fcgym_process_end_phase(void)
{
    /* This is a simplified version of end_phase() from srv_main.c */
    /* Process cities, update units, etc. */

    players_iterate(pplayer) {
        if (pplayer->is_alive && is_player_phase(pplayer, game.info.phase)) {
            /* City production and growth */
            update_city_activities(pplayer);
        }
    } players_iterate_end;
}

/*
 * Advance to the next turn.
 */
static void fcgym_advance_turn(void)
{
    /* Advance turn counter */
    game.info.turn++;

    /* Use the standard freeciv turn/phase flow */
    begin_turn(TRUE);
    begin_phase(TRUE);
}

/*
 * Check if the game is over.
 */
static bool fcgym_check_game_over(int *winner)
{
    *winner = -1;

    /* Check for various victory conditions */
    int alive_count = 0;
    struct player *last_alive = NULL;

    players_iterate(pplayer) {
        if (pplayer->is_alive && !is_barbarian(pplayer)) {
            alive_count++;
            last_alive = pplayer;
        }
    } players_iterate_end;

    /* Domination victory */
    if (alive_count == 1 && last_alive != NULL) {
        *winner = player_number(last_alive);
        return true;
    }

    /* Turn limit */
    if (game.info.turn >= game.server.end_turn) {
        /* Find highest score */
        int best_score = -1;
        players_iterate(pplayer) {
            if (pplayer->is_alive && pplayer->score.game > best_score) {
                best_score = pplayer->score.game;
                *winner = player_number(pplayer);
            }
        } players_iterate_end;
        return true;
    }

    return false;
}

/*
 * Convert tile index to coordinates (named to avoid conflict with map.h).
 */
static inline void fcgym_index_to_coords(int index, int *x, int *y)
{
    *x = index % wld.map.xsize;
    *y = index / wld.map.xsize;
}

/* ============== Public API Implementation ============== */

int fcgym_init(void)
{
    if (fcgym_initialized) {
        return 0;  /* Already initialized */
    }

    if (fcgym_init_freeciv() != 0) {
        return -1;
    }

    fcgym_initialized = true;
    return 0;
}

void fcgym_shutdown(void)
{
    if (!fcgym_initialized) {
        return;
    }

    if (fcgym_game_running) {
        /* Clean up game state - use server_game_free for full cleanup */
        server_game_free();
        fcgym_game_running = false;
    }

    fcgym_initialized = false;
}

int fcgym_new_game(const FcGameConfig *config)
{
    if (!fcgym_initialized) {
        log_error("fcgym not initialized");
        return -1;
    }

    /* Reset if a game is already running - use full server cleanup */
    if (fcgym_game_running) {
        server_game_free();
        fc_rand_uninit();
        server_game_init(FALSE);
        fcgym_game_running = false;
    }

    /* Disable aifill BEFORE loading ruleset to prevent auto-creation */
    game.info.aifill = 0;

    /* Load ruleset */
    if (fcgym_load_ruleset(config->ruleset) != 0) {
        return -1;
    }

    /* Clear any aifill players created by ruleset loading */
    /* Set aifill to 0 first to prevent more auto-creation */
    game.info.aifill = 0;
    (void) aifill(0);

    /* Set game parameters */
    game.info.fogofwar = config->fog_of_war;
    game.server.seed_setting = config->seed;

    /* IMPORTANT: Use alternating turns, not simultaneous */
    game.server.phase_mode_stored = PMT_PLAYERS_ALTERNATE;
    game.info.phase_mode = PMT_PLAYERS_ALTERNATE;

    /* Create players */
    if (fcgym_create_players(config->num_ai_players, config->ai_skill_level) != 0) {
        return -1;
    }

    /* Generate map */
    if (fcgym_generate_map(config->map_xsize, config->map_ysize, config->seed) != 0) {
        return -1;
    }

    /* Start the game */
    if (fcgym_start_game() != 0) {
        return -1;
    }

    fcgym_game_running = true;
    return 0;
}

int fcgym_reset(void)
{
    /* For now, just start a new game with the same config */
    /* TODO: Implement faster reset by saving/loading initial state */
    return -1;  /* Not yet implemented */
}

void fcgym_get_observation(FcObservation *obs)
{
    if (!fcgym_game_running || obs == NULL) {
        return;
    }

    struct player *pplayer = player_by_number(controlled_player_idx);
    if (pplayer == NULL) {
        return;
    }

    /* Map dimensions */
    obs->map_xsize = wld.map.xsize;
    obs->map_ysize = wld.map.ysize;
    obs->num_tiles = wld.map.xsize * wld.map.ysize;

    /* Game state */
    obs->turn = game.info.turn;
    obs->year = game.info.year;
    obs->phase = game.info.phase;
    obs->current_player = game.info.phase;  /* Simplified */
    obs->controlled_player = controlled_player_idx;

    /* Allocate/reallocate tile array if needed */
    if (obs->tiles == NULL) {
        obs->tiles = fc_calloc(obs->num_tiles, sizeof(FcTileObs));
    }

    /* Fill tile data */
    whole_map_iterate(&(wld.map), ptile) {
        int idx = tile_index(ptile);
        FcTileObs *tobs = &obs->tiles[idx];

        tobs->terrain = terrain_number(tile_terrain(ptile));
        tobs->owner = tile_owner(ptile) ? player_number(tile_owner(ptile)) : -1;
        tobs->has_city = tile_city(ptile) != NULL;
        tobs->has_unit = unit_list_size(ptile->units) > 0;
        tobs->visible = map_is_known(ptile, pplayer);
        tobs->explored = map_is_known(ptile, pplayer);
        tobs->extras = 0;  /* TODO: Fill in extras bitmask */
    } whole_map_iterate_end;

    /* Count units and cities for allocation */
    int total_units = 0;
    int total_cities = 0;
    players_iterate(p) {
        total_units += unit_list_size(p->units);
        total_cities += city_list_size(p->cities);
    } players_iterate_end;

    /* Allocate/reallocate unit array */
    if (obs->units == NULL || obs->max_units < total_units) {
        free(obs->units);
        obs->units = fc_calloc(total_units, sizeof(FcUnitObs));
        obs->max_units = total_units;
    }

    /* Fill unit data (only visible units) */
    obs->num_units = 0;
    players_iterate(p) {
        unit_list_iterate(p->units, punit) {
            /* Only include if visible to our player */
            if (map_is_known(punit->tile, pplayer)) {
                FcUnitObs *uobs = &obs->units[obs->num_units++];
                uobs->id = punit->id;
                uobs->type = utype_number(unit_type_get(punit));
                uobs->owner = player_number(unit_owner(punit));
                uobs->tile_index = tile_index(punit->tile);
                uobs->hp = punit->hp;
                uobs->max_hp = unit_type_get(punit)->hp;
                uobs->moves_left = punit->moves_left;
                uobs->veteran_level = punit->veteran;
                uobs->fortified = (punit->activity == ACTIVITY_FORTIFIED);
            }
        } unit_list_iterate_end;
    } players_iterate_end;

    /* Allocate/reallocate city array */
    if (obs->cities == NULL || obs->max_cities < total_cities) {
        free(obs->cities);
        obs->cities = fc_calloc(total_cities, sizeof(FcCityObs));
        obs->max_cities = total_cities;
    }

    /* Fill city data (only visible cities) */
    obs->num_cities = 0;
    players_iterate(p) {
        city_list_iterate(p->cities, pcity) {
            if (map_is_known(city_tile(pcity), pplayer)) {
                FcCityObs *cobs = &obs->cities[obs->num_cities++];
                cobs->id = pcity->id;
                cobs->owner = player_number(city_owner(pcity));
                cobs->tile_index = tile_index(city_tile(pcity));
                cobs->size = city_size_get(pcity);
                cobs->food_stock = pcity->food_stock;
                cobs->shield_stock = pcity->shield_stock;
                /* Production info */
                if (pcity->production.kind == VUT_UTYPE) {
                    cobs->producing_is_unit = true;
                    cobs->producing_type = utype_index(pcity->production.value.utype);
                } else if (pcity->production.kind == VUT_IMPROVEMENT) {
                    cobs->producing_is_unit = false;
                    cobs->producing_type = improvement_index(pcity->production.value.building);
                } else {
                    cobs->producing_type = -1;
                    cobs->producing_is_unit = false;
                }
                cobs->turns_to_complete = city_production_turns_to_build(pcity, TRUE);
            }
        } city_list_iterate_end;
    } players_iterate_end;

    /* Player info */
    obs->num_players = player_count();
    if (obs->players == NULL) {
        obs->players = fc_calloc(MAX_NUM_PLAYERS, sizeof(FcPlayerObs));
    }
    int pidx = 0;
    players_iterate(p) {
        FcPlayerObs *pobs = &obs->players[pidx++];
        pobs->index = player_number(p);
        pobs->is_alive = p->is_alive;
        pobs->is_ai = is_ai(p);
        pobs->gold = p->economic.gold;
        pobs->tax_rate = p->economic.tax;
        pobs->science_rate = p->economic.science;
        pobs->luxury_rate = p->economic.luxury;
        pobs->num_cities = city_list_size(p->cities);
        pobs->num_units = unit_list_size(p->units);
        pobs->score = p->score.game;

        /* Research info */
        struct research *presearch = research_get(p);
        if (presearch != NULL) {
            pobs->researching = presearch->researching;
            pobs->research_bulbs = presearch->bulbs_researched;
        } else {
            pobs->researching = -1;
            pobs->research_bulbs = 0;
        }
    } players_iterate_end;

    /* Game over check */
    obs->game_over = fcgym_check_game_over(&obs->winner);
}

void fcgym_free_observation(FcObservation *obs)
{
    if (obs == NULL) {
        return;
    }
    free(obs->tiles);
    free(obs->units);
    free(obs->cities);
    free(obs->players);
    memset(obs, 0, sizeof(*obs));
}

void fcgym_get_valid_actions(FcValidActions *actions)
{
    if (actions == NULL || !fcgym_game_running) {
        return;
    }

    memset(actions, 0, sizeof(*actions));

    struct player *pplayer = player_by_number(controlled_player_idx);
    if (pplayer == NULL) {
        return;
    }

    /* Can always end turn */
    actions->can_end_turn = true;

    /* Count units and cities for allocation */
    int num_units = unit_list_size(pplayer->units);
    int num_cities = city_list_size(pplayer->cities);

    /* Allocate unit actions array */
    if (num_units > 0) {
        actions->unit_actions = fc_calloc(num_units, sizeof(*actions->unit_actions));
        actions->num_unit_actions = num_units;

        int idx = 0;
        unit_list_iterate(pplayer->units, punit) {
            actions->unit_actions[idx].unit_id = punit->id;

            /* Check movement in each valid direction */
            for (int d = 0; d < wld.map.num_valid_dirs; d++) {
                enum direction8 dir = wld.map.valid_dirs[d];
                struct tile *dst_tile = mapstep(&(wld.map), punit->tile, dir);

                if (dst_tile != NULL && punit->moves_left > 0) {
                    /* Check if can move to tile (simplified - just check basic movement) */
                    if (unit_can_move_to_tile(&(wld.map), punit, dst_tile, FALSE, FALSE, FALSE)) {
                        actions->unit_actions[idx].can_move[dir] = true;
                    }
                    /* Also allow if there's an enemy to attack */
                    if (is_enemy_unit_tile(dst_tile, pplayer) ||
                        is_enemy_city_tile(dst_tile, pplayer)) {
                        actions->unit_actions[idx].can_attack = true;
                        actions->unit_actions[idx].can_move[dir] = true;
                    }
                }
            }

            /* Check fortify capability */
            if (can_unit_do_activity(&(wld.map), punit, ACTIVITY_FORTIFYING,
                                     activity_default_action(ACTIVITY_FORTIFYING))) {
                actions->unit_actions[idx].can_fortify = true;
            }

            /* Check can build city */
            if (is_action_enabled_unit_on_tile(
                    &(wld.map), ACTION_FOUND_CITY, punit, punit->tile, NULL)) {
                actions->unit_actions[idx].can_build_city = true;
            }

            /* Check can build road/irrigation/mine */
            /* For roads, we need to find a target extra first */
            {
                struct extra_type *road_target = next_extra_for_tile(
                    punit->tile, EC_ROAD, unit_owner(punit), punit);
                if (road_target != NULL &&
                    can_unit_do_activity_targeted(&(wld.map), punit, ACTIVITY_GEN_ROAD,
                                                  activity_default_action(ACTIVITY_GEN_ROAD),
                                                  road_target)) {
                    actions->unit_actions[idx].can_build_road = true;
                }
            }
            if (can_unit_do_activity(&(wld.map), punit, ACTIVITY_IRRIGATE,
                                     activity_default_action(ACTIVITY_IRRIGATE))) {
                actions->unit_actions[idx].can_build_irrigation = true;
            }
            if (can_unit_do_activity(&(wld.map), punit, ACTIVITY_MINE,
                                     activity_default_action(ACTIVITY_MINE))) {
                actions->unit_actions[idx].can_build_mine = true;
            }

            /* Check if unit can be disbanded */
            actions->unit_actions[idx].can_disband =
                unit_can_do_action(punit, ACTION_DISBAND_UNIT) &&
                is_action_enabled_unit_on_self(&(wld.map), ACTION_DISBAND_UNIT, punit);

            idx++;
        } unit_list_iterate_end;
    }

    /* Allocate city actions array */
    if (num_cities > 0) {
        actions->city_actions = fc_calloc(num_cities, sizeof(*actions->city_actions));
        actions->num_city_actions = num_cities;

        int cidx = 0;
        city_list_iterate(pplayer->cities, pcity) {
            actions->city_actions[cidx].city_id = pcity->id;

            /* Count buildable units */
            int num_buildable_units = 0;
            unit_type_iterate(ptype) {
                if (can_city_build_unit_now(&(wld.map), pcity, ptype)) {
                    num_buildable_units++;
                }
            } unit_type_iterate_end;

            /* Allocate and fill buildable units */
            if (num_buildable_units > 0) {
                actions->city_actions[cidx].buildable_units =
                    fc_malloc(num_buildable_units * sizeof(int));
                actions->city_actions[cidx].num_buildable_units = num_buildable_units;

                int uidx = 0;
                unit_type_iterate(ptype) {
                    if (can_city_build_unit_now(&(wld.map), pcity, ptype)) {
                        actions->city_actions[cidx].buildable_units[uidx++] =
                            utype_index(ptype);
                    }
                } unit_type_iterate_end;
            }

            /* Count buildable buildings */
            int num_buildable_buildings = 0;
            improvement_iterate(pimprove) {
                if (can_city_build_improvement_now(pcity, pimprove)) {
                    num_buildable_buildings++;
                }
            } improvement_iterate_end;

            /* Allocate and fill buildable buildings */
            if (num_buildable_buildings > 0) {
                actions->city_actions[cidx].buildable_buildings =
                    fc_malloc(num_buildable_buildings * sizeof(int));
                actions->city_actions[cidx].num_buildable_buildings = num_buildable_buildings;

                int bidx = 0;
                improvement_iterate(pimprove) {
                    if (can_city_build_improvement_now(pcity, pimprove)) {
                        actions->city_actions[cidx].buildable_buildings[bidx++] =
                            improvement_index(pimprove);
                    }
                } improvement_iterate_end;
            }

            /* Check if can buy current production */
            /* Must match conditions in really_handle_city_buy() */
            actions->city_actions[cidx].can_buy =
                (pcity->turn_founded != game.info.turn &&  /* Not founded this turn */
                 !pcity->did_buy &&                        /* Haven't bought this turn */
                 pcity->shield_stock < city_production_build_shield_cost(pcity) &&
                 pplayer->economic.gold >= city_production_buy_gold_cost(pcity) &&
                 /* Can't buy units when in anarchy */
                 (pcity->production.kind != VUT_UTYPE || pcity->anarchy == 0));

            cidx++;
        } city_list_iterate_end;
    }

    /* Get researchable techs - only those with prerequisites known */
    struct research *presearch = research_get(pplayer);
    if (presearch != NULL) {
        /* Count researchable techs (prereqs known, not already known) */
        int num_techs = 0;
        advance_iterate(adv) {
            if (research_invention_state(presearch, advance_index(adv)) == TECH_PREREQS_KNOWN) {
                num_techs++;
            }
        } advance_iterate_end;

        /* Allocate and fill researchable techs */
        if (num_techs > 0) {
            actions->researchable_techs = fc_malloc(num_techs * sizeof(int));
            actions->num_researchable_techs = num_techs;

            int tidx = 0;
            advance_iterate(adv) {
                if (research_invention_state(presearch, advance_index(adv)) == TECH_PREREQS_KNOWN) {
                    actions->researchable_techs[tidx++] = advance_index(adv);
                }
            } advance_iterate_end;
        }
    }
}

void fcgym_free_valid_actions(FcValidActions *actions)
{
    if (actions == NULL) {
        return;
    }

    /* Free unit actions arrays */
    if (actions->unit_actions != NULL) {
        free(actions->unit_actions);
    }

    /* Free city actions arrays */
    if (actions->city_actions != NULL) {
        for (int i = 0; i < actions->num_city_actions; i++) {
            free(actions->city_actions[i].buildable_units);
            free(actions->city_actions[i].buildable_buildings);
        }
        free(actions->city_actions);
    }

    /* Free research array */
    free(actions->researchable_techs);

    memset(actions, 0, sizeof(*actions));
}

FcStepResult fcgym_step(const FcAction *action)
{
    FcStepResult result = {0};

    if (!fcgym_game_running || action == NULL) {
        result.info = "Game not running or invalid action";
        return result;
    }

    struct player *pplayer = player_by_number(controlled_player_idx);
    if (pplayer == NULL) {
        result.info = "Controlled player not found";
        return result;
    }

    /* Execute the action */
    switch (action->type) {
    case FCGYM_ACTION_UNIT_MOVE: {
        struct unit *punit = game_unit_by_number(action->actor_id);
        if (punit != NULL && unit_owner(punit) == pplayer) {
            enum direction8 dir = action->sub_target;
            struct tile *dst_tile = mapstep(&(wld.map), punit->tile, dir);
            if (dst_tile != NULL) {
                /* Use unit_move_handling - it calls unit_perform_action internally
                 * and handles edge cases like transport embark.
                 * TRUE skips action decision dialogs (like AI/goto does). */
                unit_move_handling(punit, dst_tile, TRUE);
            }
        }
        break;
    }

    case FCGYM_ACTION_UNIT_ATTACK: {
        struct unit *punit = game_unit_by_number(action->actor_id);
        if (punit != NULL && unit_owner(punit) == pplayer) {
            /* Target is a tile index containing enemy units */
            struct tile *target_tile = index_to_tile(&(wld.map), action->target_id);
            if (target_tile != NULL) {
                /* Check if attack is possible first */
                if (is_action_enabled_unit_on_stack(&(wld.map), ACTION_ATTACK,
                                                    punit, target_tile)) {
                    /* Use unit_perform_action with ACTION_ATTACK */
                    unit_perform_action(pplayer, punit->id, tile_index(target_tile),
                                       NO_TARGET, "", ACTION_ATTACK, ACT_REQ_RULES);
                }
            }
        }
        break;
    }

    case FCGYM_ACTION_UNIT_FORTIFY: {
        struct unit *punit = game_unit_by_number(action->actor_id);
        if (punit != NULL && unit_owner(punit) == pplayer) {
            /* Use unit_activity_handling - the proper high-level handler */
            unit_activity_handling(punit, ACTIVITY_FORTIFYING,
                                   activity_default_action(ACTIVITY_FORTIFYING));
        }
        break;
    }

    case FCGYM_ACTION_UNIT_BUILD_CITY: {
        struct unit *punit = game_unit_by_number(action->actor_id);
        if (punit != NULL && unit_owner(punit) == pplayer) {
            /* Get a suggested city name */
            const char *name = city_name_suggestion(pplayer, punit->tile);
            /* Use action system to build city */
            unit_perform_action(pplayer, punit->id, tile_index(punit->tile),
                               0, name, ACTION_FOUND_CITY, ACT_REQ_PLAYER);
        }
        break;
    }

    case FCGYM_ACTION_UNIT_BUILD_ROAD: {
        struct unit *punit = game_unit_by_number(action->actor_id);
        if (punit != NULL && unit_owner(punit) == pplayer) {
            handle_unit_change_activity(pplayer, punit->id,
                                        ACTIVITY_GEN_ROAD, action->sub_target);
        }
        break;
    }

    case FCGYM_ACTION_UNIT_BUILD_IRRIGATION: {
        struct unit *punit = game_unit_by_number(action->actor_id);
        if (punit != NULL && unit_owner(punit) == pplayer) {
            handle_unit_change_activity(pplayer, punit->id,
                                        ACTIVITY_IRRIGATE, action->sub_target);
        }
        break;
    }

    case FCGYM_ACTION_UNIT_BUILD_MINE: {
        struct unit *punit = game_unit_by_number(action->actor_id);
        if (punit != NULL && unit_owner(punit) == pplayer) {
            handle_unit_change_activity(pplayer, punit->id,
                                        ACTIVITY_MINE, action->sub_target);
        }
        break;
    }

    case FCGYM_ACTION_UNIT_DISBAND: {
        struct unit *punit = game_unit_by_number(action->actor_id);
        if (punit != NULL && unit_owner(punit) == pplayer) {
            /* Target is the unit itself for disband actions */
            unit_perform_action(pplayer, punit->id, punit->id,
                               0, NULL, ACTION_DISBAND_UNIT, ACT_REQ_PLAYER);
        }
        break;
    }

    case FCGYM_ACTION_CITY_BUILD: {
        struct city *pcity = game_city_by_number(action->actor_id);
        if (pcity != NULL && city_owner(pcity) == pplayer) {
            /* Use handle_city_change - the proper high-level handler
             * sub_target: 0 = unit, 1 = building
             * target_id: unit type index or improvement index */
            int production_kind = action->sub_target ? VUT_IMPROVEMENT : VUT_UTYPE;
            handle_city_change(pplayer, pcity->id, production_kind, action->target_id);
        }
        break;
    }

    case FCGYM_ACTION_CITY_BUY: {
        struct city *pcity = game_city_by_number(action->actor_id);
        if (pcity != NULL && city_owner(pcity) == pplayer) {
            /* Try to buy current production */
            really_handle_city_buy(pplayer, pcity);
        }
        break;
    }

    case FCGYM_ACTION_RESEARCH_SET: {
        /* target_id is the tech index to research */
        handle_player_research(pplayer, action->target_id);
        break;
    }

    case FCGYM_ACTION_END_TURN: {
        /* Mark player as done with phase */
        pplayer->phase_done = TRUE;

        /* Run AI for their phase */
        fcgym_run_ai_phase();

        /* Process end of phase */
        fcgym_process_end_phase();

        /* Advance turn */
        fcgym_advance_turn();

        /* Check for game over */
        int winner;
        if (fcgym_check_game_over(&winner)) {
            result.done = true;
            if (winner == controlled_player_idx) {
                result.reward = 1.0f;
            } else if (winner >= 0) {
                result.reward = -1.0f;
            }
        }
        break;
    }

    case FCGYM_ACTION_NOOP:
        /* Do nothing */
        break;

    default:
        result.info = "Unknown action type";
        break;
    }

    /* Calculate reward based on score change */
    /* TODO: Implement proper reward calculation */

    return result;
}

int fcgym_num_unit_types(void)
{
    return utype_count();
}

int fcgym_num_building_types(void)
{
    return improvement_count();
}

int fcgym_num_techs(void)
{
    return advance_count();
}

const char* fcgym_unit_type_name(int index)
{
    struct unit_type *ut = utype_by_number(index);
    return ut ? utype_rule_name(ut) : NULL;
}

const char* fcgym_building_type_name(int index)
{
    struct impr_type *pimprove = improvement_by_number(index);
    return pimprove ? improvement_rule_name(pimprove) : NULL;
}

const char* fcgym_tech_name(int index)
{
    struct advance *padvance = advance_by_number(index);
    return padvance ? advance_rule_name(padvance) : NULL;
}

struct player* fcgym_get_controlled_player(void)
{
    if (!fcgym_game_running) {
        return NULL;
    }
    return player_by_number(controlled_player_idx);
}

struct unit* fcgym_get_unit(int unit_id)
{
    if (!fcgym_game_running) {
        return NULL;
    }
    return game_unit_by_number(unit_id);
}

struct city* fcgym_get_city(int city_id)
{
    if (!fcgym_game_running) {
        return NULL;
    }
    return game_city_by_number(city_id);
}

struct tile* fcgym_get_tile(int x, int y)
{
    if (!fcgym_game_running) {
        return NULL;
    }
    return map_pos_to_tile(&(wld.map), x, y);
}
