import gymnasium as gym
import ale_py
import itertools
import os
import gc

from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import AtariWrapper
from stable_baselines3.common.vec_env import DummyVecEnv

# Registrar Atari
gym.register_envs(ale_py)

env_name = "ALE/PrivateEye-v5"


def make_env():
    env = gym.make(env_name)
    env = AtariWrapper(env)
    return env


# Espacio de búsqueda de hiperparámetros (optimizado para 8GB RAM)
learning_rates = [0.00005, 0.0001, 0.0002]
batch_sizes = [32]  # reducido para ahorrar memoria
buffer_sizes = [30000, 50000]  # menor consumo de RAM


experiments = list(itertools.product(learning_rates, batch_sizes, buffer_sizes))

print("Total experiments:", len(experiments))


# Crear carpetas si no existen
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)


for i, (lr, batch, buffer) in enumerate(experiments):

    print("\n==============================")
    print(f"Running experiment {i+1}")
    print("learning_rate:", lr)
    print("batch_size:", batch)
    print("buffer_size:", buffer)
    print("==============================")

    env = DummyVecEnv([make_env])

    model = DQN(
        "CnnPolicy",
        env,
        learning_rate=lr,
        batch_size=batch,
        buffer_size=buffer,
        learning_starts=10000,
        gamma=0.99,
        train_freq=4,
        target_update_interval=2000,
        exploration_fraction=0.2,
        exploration_final_eps=0.01,
        verbose=1,
        tensorboard_log=f"./logs/exp_{i+1}/"
    )

    # entrenamiento (reducido para evitar crash de RAM)
    model.learn(total_timesteps=500000)

    model_path = f"models/dqn_privateeye_exp{i+1}"

    model.save(model_path)

    print("Model saved:", model_path)

    # liberar memoria antes del siguiente experimento
    env.close()
    del model
    del env

    gc.collect()

print("\nAll experiments completed")