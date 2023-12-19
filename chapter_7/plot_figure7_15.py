import plot_lib


if __name__ == '__main__':
    experiments = [
        dict(
            agent_id='naive_deep_q',
            env_name='cartpole',
            base_path='./logs/naive_deep_q/cartpole/',
        ),
        dict(
            agent_id='deep_q_replay',
            env_name='cartpole',
            base_path='./logs/deep_q_replay/cartpole/',
        ),
        dict(
            agent_id='deep_q_targetnet',
            env_name='cartpole',
            base_path='./logs/deep_q_targetnet/cartpole/',
        ),
        dict(
            agent_id='dqn',
            env_name='cartpole',
            base_path='./logs/dqn/cartpole/',
        ),
    ]

    plot_lib.plot_and_save_experiments(experiments)
