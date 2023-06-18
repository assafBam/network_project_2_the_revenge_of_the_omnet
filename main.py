import Environment
import stable_baselines3 as st
import numpy as np
from stable_baselines3.common import callbacks
from stable_baselines3.common.noise import NormalActionNoise

if __name__ == '__main__':
    # train the model

    envi = Environment.drl_agent_env()
    n_actions = envi.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    policy_kwargs = dict(net_arch=[91, 42])

    model = st.DDPG(policy='MlpPolicy', env=envi,
                    gamma=0.99,
                    buffer_size=1600,
                    action_noise=action_noise,
                    learning_starts=1,
                    batch_size=16,
                    learning_rate=0.00093,
                    train_freq=(1, 'episode'),
                    gradient_steps=-1,
                    tensorboard_log='log',
                    policy_kwargs=policy_kwargs)

    eval_callback = callbacks.EvalCallback(envi, best_model_save_path='./logss/',
                                           log_path='./logss/', eval_freq=7000,
                                           deterministic=True, render=False, n_eval_episodes=10)

    model.learn(total_timesteps=70000, callback=eval_callback)

    model.save("my_model")
