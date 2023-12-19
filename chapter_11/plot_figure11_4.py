import plot_lib


if __name__ == '__main__':
    experiments = [
        dict(
            agent_id='ppo',
            env_name='humanoid',
            base_path='./logs/ppo/humanoid/',
        ),
    ]

    plot_lib.plot_and_save_experiments(experiments, show_eval_data=False)
