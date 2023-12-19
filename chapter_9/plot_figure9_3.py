import plot_lib


if __name__ == '__main__':
    experiments = [
        dict(
            agent_id='reinforce',
            env_name='cartpole',
            base_path='./logs/reinforce/cartpole/',
        ),
        dict(
            agent_id='reinforce_baseline',
            env_name='cartpole',
            base_path='./logs/reinforce_baseline/cartpole/',
        ),
    ]

    plot_lib.plot_and_save_experiments(experiments)
