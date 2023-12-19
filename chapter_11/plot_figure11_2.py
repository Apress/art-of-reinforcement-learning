import plot_lib


if __name__ == '__main__':
    experiments = [
        dict(
            agent_id='ppo',
            env_name='pong',
            base_path='./logs/ppo/pong/',
        ),
    ]

    plot_lib.plot_and_save_experiments(experiments)
