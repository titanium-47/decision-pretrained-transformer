from itertools import combinations
import numpy as np


from envs.darkroom_env import DarkroomEnvVec, DarkroomEnv
from envs.keydoor_env import KeyDoorVecEnv, KeyDoorEnv
from envs.navigation_env import NavigationVecEnv, NavigationEnv


def create_darkroom_env(env_name, dataset_size, n_envs):
    if env_name == "darkroom-easy":
        dim = 10
        environment_horizon = 100
    elif env_name == "darkroom-hard":
        dim = 25
        environment_horizon = 200
    goals = np.array([[(j, i) for i in range(dim)] for j in range(dim)]).reshape(-1, 2)
    np.random.RandomState(seed=0).shuffle(goals)
    train_test_split = int(0.8 * len(goals))
    train_goals = goals[:train_test_split]
    test_goals = goals[train_test_split:]
    factor = max(1, int(100 // len(test_goals)))

    eval_goals = np.array(test_goals.tolist() * factor)
    train_goals = np.repeat(train_goals, dataset_size // (dim * dim), axis=0)
    test_goals = np.repeat(test_goals, dataset_size // (dim * dim), axis=0)

    train_envs = [DarkroomEnv(dim, goal, environment_horizon) for goal in train_goals]
    eval_envs = [DarkroomEnv(dim, goal, environment_horizon) for goal in eval_goals]
    test_envs = [DarkroomEnv(dim, goal, environment_horizon) for goal in test_goals]

    train_envs = [
        DarkroomEnvVec(train_envs[i : i + n_envs])
        for i in range(0, len(train_envs), n_envs)
    ]
    test_envs = [
        DarkroomEnvVec(test_envs[i : i + n_envs])
        for i in range(0, len(test_envs), n_envs)
    ]
    eval_envs = [
        DarkroomEnvVec(eval_envs[i : i + n_envs])
        for i in range(0, len(eval_envs), n_envs)
    ]
    return train_envs, test_envs, eval_envs


def create_keydoor_env(env_name, dataset_size, n_envs):
    dim = 5
    environment_horizon = 50

    markovian = "markovian" in env_name
    locations = np.array([[(j, i) for i in range(dim)] for j in range(dim)]).reshape(
        -1, 2
    )
    location_idxs = np.arange(len(locations))
    key_door_pairs = np.array(list(combinations(location_idxs, 2)))
    np.random.RandomState(seed=0).shuffle(key_door_pairs)
    train_test_split = int(0.8 * len(key_door_pairs))
    train_goals = key_door_pairs[:train_test_split]
    test_goals = key_door_pairs[train_test_split:]

    factor = max(1, int(100 // len(test_goals)))
    eval_goals = np.array(test_goals.tolist() * factor)
    train_goals = np.repeat(train_goals, dataset_size // len(key_door_pairs), axis=0)
    test_goals = np.repeat(test_goals, dataset_size // len(key_door_pairs), axis=0)

    train_envs = [
        KeyDoorEnv(dim, locations[key], locations[door], environment_horizon, markovian)
        for key, door in train_goals
    ]
    eval_envs = [
        KeyDoorEnv(dim, locations[key], locations[door], environment_horizon, markovian)
        for key, door in eval_goals
    ]
    test_envs = [
        KeyDoorEnv(dim, locations[key], locations[door], environment_horizon, markovian)
        for key, door in test_goals
    ]

    train_envs = [
        KeyDoorVecEnv(train_envs[i : i + n_envs])
        for i in range(0, len(train_envs), n_envs)
    ]
    test_envs = [
        KeyDoorVecEnv(test_envs[i : i + n_envs])
        for i in range(0, len(test_envs), n_envs)
    ]
    eval_envs = [
        KeyDoorVecEnv(eval_envs[i : i + n_envs])
        for i in range(0, len(eval_envs), n_envs)
    ]
    return train_envs, test_envs, eval_envs


def create_navigation_env(env_name, dataset_size, n_envs):
    radius = 2.0
    environment_horizon = 50
    dense_reward = "dense" in env_name

    # goals in a semicircle based on radius
    angles = np.linspace(0, np.pi, 100)
    goals = np.array(
        [[radius * np.cos(angle), radius * np.sin(angle)] for angle in angles]
    )
    np.random.RandomState(seed=0).shuffle(goals)
    train_test_split = int(0.8 * len(goals))
    train_goals = goals[:train_test_split]
    test_goals = goals[train_test_split:]

    factor = max(1, int(100 // len(test_goals)))
    eval_goals = np.array(test_goals.tolist() * factor)
    train_goals = np.repeat(train_goals, dataset_size // len(goals), axis=0)
    test_goals = np.repeat(test_goals, dataset_size // len(goals), axis=0)

    train_envs = [
        NavigationEnv(radius, goal, environment_horizon, dense_reward)
        for goal in train_goals
    ]
    eval_envs = [
        NavigationEnv(radius, goal, environment_horizon, dense_reward)
        for goal in eval_goals
    ]
    test_envs = [
        NavigationEnv(radius, goal, environment_horizon, dense_reward)
        for goal in test_goals
    ]

    train_envs = [
        NavigationVecEnv(train_envs[i : i + n_envs])
        for i in range(0, len(train_envs), n_envs)
    ]
    test_envs = [
        NavigationVecEnv(test_envs[i : i + n_envs])
        for i in range(0, len(test_envs), n_envs)
    ]
    eval_envs = [
        NavigationVecEnv(eval_envs[i : i + n_envs])
        for i in range(0, len(eval_envs), n_envs)
    ]
    return train_envs, test_envs, eval_envs


def create_env(env_name, dataset_size, n_envs):
    if "darkroom" in env_name:
        return create_darkroom_env(env_name, dataset_size, n_envs)
    elif "keydoor" in env_name:
        return create_keydoor_env(env_name, dataset_size, n_envs)
    elif "navigation" in env_name:
        return create_navigation_env(env_name, dataset_size, n_envs)
    else:
        raise ValueError(f"Unknown environment name: {env_name}")


def test_all_envs():
    # generate a single env and see if taking the optimal action gets to the goal
    envs = [
        DarkroomEnv(10, (9, 9), 100),
        KeyDoorEnv(5, (0, 0), (4, 4), 50, markovian=False),
        KeyDoorEnv(5, (0, 0), (4, 4), 50, markovian=True),
        NavigationEnv(2.0, (1.0, 1.0), 50, dense_reward=False),
        NavigationEnv(2.0, (1.0, 1.0), 50, dense_reward=True),
    ]
    for env in envs:
        env_name = env.__class__.__name__
        print(f"Testing environment: {env_name}")
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            if "KeyDoor" in env_name:
                action = env.opt_action(obs, env.have_key)
            else:
                action = env.opt_action(obs)
            obs, reward, done, _ = env.step(action)
            total_reward += reward
        print(f"Total reward: {total_reward}")
        assert total_reward > 0, f"Optimal policy failed in {env_name}"


if __name__ == "__main__":
    test_all_envs()
