import plot_lib


if __name__ == '__main__':
    experiments = [
        dict(
            agent_id='actor_critic',
            env_name='pong',
            base_path='./logs/actor_critic_entropy/pong/',
        ),
    ]

    plot_lib.plot_and_save_experiments(experiments)
