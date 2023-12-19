import plot_lib


if __name__ == '__main__':
    experiments = [
        dict(
            agent_id='actor_critic',
            env_name='humanoid',
            base_path='./logs/actor_critic/humanoid/',
        ),
    ]

    plot_lib.plot_and_save_experiments(experiments)
