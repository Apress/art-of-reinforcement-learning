import plot_lib


if __name__ == '__main__':
    experiments = [
        dict(
            agent_id='dqn',
            env_name='pong',
            base_path='./logs/dqn/pong/',
        ),
    ]

    plot_lib.plot_and_save_experiments(experiments)
