import plot_lib


if __name__ == '__main__':
    experiments = [
        dict(
            agent_id='ppo_4actors',
            env_name='humanoid',
            base_path='./logs/ppo/humanoid/4',
        ),
        dict(
            agent_id='ppo_8actors',
            env_name='humanoid',
            base_path='./logs/ppo/humanoid/8',
        ),
        dict(
            agent_id='ppo_16actors',
            env_name='humanoid',
            base_path='./logs/ppo/humanoid/16',
        ),
    ]

    plot_lib.plot_and_save_experiments(experiments, show_eval_data=False)
