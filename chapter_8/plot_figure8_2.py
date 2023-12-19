import plot_lib


if __name__ == '__main__':
    experiments = [
        dict(
            agent_id='prioritized_dqn',
            env_name='riverraid',
            base_path='./logs/prioritized_dqn/riverraid/',
        ),
    ]

    plot_lib.plot_and_save_experiments(experiments)
