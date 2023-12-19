import plot_lib


if __name__ == '__main__':
    experiments = [
        dict(
            agent_id='naive_deep_q',
            env_name='mountaincar',
            base_path='./logs/naive_deep_q/mountaincar/',
        ),
        dict(
            agent_id='deep_q_replay',
            env_name='mountaincar',
            base_path='./logs/deep_q_replay/mountaincar/',
        ),
        dict(
            agent_id='deep_q_targetnet',
            env_name='mountaincar',
            base_path='./logs/deep_q_targetnet/mountaincar/',
        ),
        dict(
            agent_id='dqn',
            env_name='mountaincar',
            base_path='./logs/dqn/mountaincar/',
        ),
    ]

    plot_lib.plot_and_save_experiments(experiments)
