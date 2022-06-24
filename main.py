"""
Example
"""
import copy
from atcenv.env import ExperienceReplay

WANDB_USAGE = True

if __name__ == "__main__":
    import random

    random.seed(42)
    from jsonargparse import ArgumentParser, ActionConfigFile
    from atcenv import Environment
    from tqdm import tqdm
    import collections
    from atcenv.definitions import *
    from atcenv.policy import *
    import statistics

    # Policy constants:
    EPISODES = 500
    CONFLICT_DISTANCE = 5.

    parser = ArgumentParser(
        prog='Conflict resolution environment',
        description='Basic conflict resolution environment for training policies with reinforcement learning',
        print_config='--print_config',
        parser_mode='yaml'
    )
    parser.add_argument('--episodes', type=int, default=EPISODES)
    parser.add_argument('--config', action=ActionConfigFile)
    parser.add_class_arguments(Environment, 'env')
    # parse arguments
    args = parser.parse_args()

    # init environment
    env = Environment(**vars(args.env))
    comparison_env = Environment(**vars(args.env))

    # Policy parameters: Change it at env.py
    NUM_FLIGHTS = env.num_flights
    ALERT_DISTANCE = env.alert_distance
    ALERT_TIME = env.alert_time
    PERFORMANCE_LIMITATION = env.performance_limitation

    # wandb usage
    if WANDB_USAGE:
        import wandb
        wandb.init(project="(Lidia) Policy - RESULTS", group='/Comparison run', entity="emc-upc",
                   name='Episodes=500, Num_Flights=10, Alert_distance=10.NM, Alert_time=70s, Performance_limitation=180º')
        wandb.config.update({"Episodes": EPISODES, "Number of flights": NUM_FLIGHTS, "Alert distance": ALERT_DISTANCE,
                             "Alert time": ALERT_TIME, "Conflict distance": CONFLICT_DISTANCE, "Performance limitation": PERFORMANCE_LIMITATION})

        wandb.log({'Conflict 5NM': 5})

    # Short memory definition
    shortMemory = collections.namedtuple('ShortMemory', field_names=['obs', 'action'])

    # Prioritized replay buffer
    max_memory_size = 10
    do_nothing = [0] * comparison_env.num_flights
    short_memory_size = 2
    replay_buffer = PrioritizedReplayBuffer(maxlen=max_memory_size)
    print('*** Filling the Replay Buffer ***')
    for e in tqdm(range(max_memory_size)):
        """
        Filling the Replay Buffer
        """
        obs = env.reset()
        done = False
        short_memo = ExperienceReplay(short_memory_size)
        short_exp = shortMemory(env.distances_matrix(), do_nothing)
        short_memo.append(short_exp)

    # run episodes
    vector_conflicts_reduction = []
    vector_conflicts_reduction_NoPolicy = []
    vector_alerts_reduction = []
    vector_alerts_reduction_NoPolicy = []
    vector_extra_distance = []

    for e in tqdm(range(args.episodes)):
        obs, state = env.reset()
        c_obs = comparison_env.comparison_reset(state)
        done = False

        short_memo = ExperienceReplay(short_memory_size)
        short_exp = shortMemory(env.distances_matrix(), do_nothing)
        short_memo.append(short_exp)

        while not done:
            previous_distances = env.distances_matrix()
            actions = policy_action(short_memo, env)
            done, info = env.step(actions)
            # env.render()

            c_done, c_info = comparison_env.step(do_nothing)
            # comparison_env.render()

            if WANDB_USAGE:
                conflicts_reduction = 0
                if comparison_env.n_conflicts_step == 0:
                    if env.n_conflicts_step == 0:
                        conflicts_reduction = 100
                    else:
                        conflicts_reduction = 0
                else:
                    if env.n_conflicts_step == 0:
                        conflicts_reduction = 100
                    else:
                        conflicts_reduction = ((comparison_env.n_conflicts_step - env.n_conflicts_step) / comparison_env.n_conflicts_step) * 100

                wandb.log({'Number of conflicts each timestep with policy': env.n_conflicts_step,
                           'Number of conflicts each timestep without policy': comparison_env.n_conflicts_step,
                           'Number of alerts each timestep with policy': env.n_alerts_step,
                           'Number of alerts each timestep without policy': comparison_env.n_alerts_step,
                           'Conflicts reduction each step (%)': conflicts_reduction,
                           'ATC instructions each step': env.ATC_instructions_step})

            # time.sleep(0.05)
            # adding the action to a memory that will help us to get a better human policy.
            short_exp = shortMemory(previous_distances, actions)
            short_memo.append(short_exp)

        if WANDB_USAGE:
            # compute extra distance
            extra_distance_per_episode_m = 0.
            extra_distance_per_episode_NM = 0.
            for i, f in enumerate(env.flights):
                extra_distance = (f.traveled_distance + f.position.distance(f.target)) - f.initial_distance
                # get the sum of all agents extra distance
                extra_distance_per_episode_m += extra_distance
                extra_distance_per_episode_NM = extra_distance_per_episode_m * u.m

            vector_extra_distance.append(extra_distance_per_episode_NM)

            # Compute conflicts per episode and maximum ATC instructions per flight during all episode
            n_real_conflicts_episode = 0
            n_real_conflicts_episode_without_policy = 0
            n_real_alerts_episode = 0
            n_real_alerts_episode_without_policy = 0
            ATC_instructions_per_flight = 0
            for i in range(env.num_flights):
                ATC_instructions_per_flight = max(ATC_instructions_per_flight, env.ATC_instructions_per_flight[i])
                for j in range(env.num_flights):
                    # conflicts
                    if env.matrix_real_conflicts_episode[i, j]:
                        n_real_conflicts_episode += 1
                    if comparison_env.matrix_real_conflicts_episode[i, j]:
                        n_real_conflicts_episode_without_policy += 1
                    # alerts
                    if env.matrix_real_alerts_episode[i, j]:
                        n_real_alerts_episode += 1
                    if comparison_env.matrix_real_alerts_episode[i, j]:
                        n_real_alerts_episode_without_policy += 1

            vector_conflicts_reduction.append(n_real_conflicts_episode)
            vector_conflicts_reduction_NoPolicy.append(n_real_conflicts_episode_without_policy)
            vector_alerts_reduction.append(n_real_alerts_episode)
            vector_alerts_reduction_NoPolicy.append(n_real_alerts_episode_without_policy)

            wandb.log({'Sum of number of conflicts each timestep during all episode with policy': env.n_conflicts_episode,
                       'Sum of number of conflicts each timestep during all episode without policy': comparison_env.n_conflicts_episode,
                       'Sum of number of alerts each timestep during all episode with policy': env.n_alerts_episode,
                       'Sum of number of alerts each timestep during all episode without policy': comparison_env.n_alerts_episode,
                       'Conflicts each episode with policy': n_real_conflicts_episode,
                       'Conflicts each episode without policy': n_real_conflicts_episode_without_policy,
                       'Alerts each episode with policy': n_real_alerts_episode,
                       'Alerts each episode without policy': n_real_alerts_episode_without_policy,
                       'Sum of ATC instructions each episode': env.ATC_instructions_episode,
                       'Maximum ATC instructions per one flight each episode': ATC_instructions_per_flight,
                       'Extra distance per episode [meters]': extra_distance_per_episode_m,
                       'Extra distance per episode [NM]': extra_distance_per_episode_NM,
                       'Minimum separation distance each episode with policy': min(env.critical_distance),
                       'Minimum separation distance each episode without policy': min(comparison_env.critical_distance)})

    """ Outside the episodes loop"""
    if WANDB_USAGE:
        # Computing mean reduction conflicts
        mean_conflicts = statistics.mean(vector_conflicts_reduction)
        mean_conflicts_NoPolicy = statistics.mean(vector_conflicts_reduction_NoPolicy)
        if mean_conflicts_NoPolicy == 0:
            if mean_conflicts == 0:
                mean_reduction_conflicts = 100
            else:
                mean_reduction_conflicts = 0
        else:
            if mean_conflicts == 0:
                mean_reduction_conflicts = 100
            else:
                mean_reduction_conflicts = ((mean_conflicts_NoPolicy - mean_conflicts) / mean_conflicts_NoPolicy) * 100
        # Computing mean reduction alerts
        mean_alerts = statistics.mean(vector_alerts_reduction)
        mean_alerts_NoPolicy = statistics.mean(vector_alerts_reduction_NoPolicy)
        if mean_alerts_NoPolicy == 0:
            if mean_alerts == 0:
                mean_reduction_alerts = 100
            else:
                mean_reduction_alerts = 0
        else:
            if mean_alerts == 0:
                mean_reduction_alerts = 100
            else:
                mean_reduction_alerts = ((mean_alerts_NoPolicy - mean_alerts) / mean_alerts_NoPolicy) * 100
        # Computing mean extra distance
        mean_extra_distance = statistics.mean(vector_extra_distance)

        wandb.log({'Mean reduction conflicts (%)': mean_reduction_conflicts,
                   'Mean reduction alerts (%)': mean_reduction_alerts,
                   'Mean extra distance [NM]': mean_extra_distance,
                   'Conflict 5NM': 5})

        env.close()
        comparison_env.close()
