import plot_lib


if __name__ == '__main__':
    experiments = [
        dict(
            agent_id='naive_deep_q',
            env_name='cartpole',
            base_path='./logs/naive_deep_q/cartpole/',
        ),
    ]

    plot_lib.plot_and_save_experiments(experiments)
