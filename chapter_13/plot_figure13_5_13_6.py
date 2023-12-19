import plot_lib


if __name__ == '__main__':
    experiments = [
        dict(
            agent_id='rnd_ppo_32actors',
            env_name='montezumarevenge',
            base_path='./logs/rnd_ppo/montezumarevenge',
        ),
    ]

    plot_lib.plot_and_save_experiments(
        experiments, additional_columns=['train_episode_visited_rooms', 'eval_episode_visited_rooms']
    )
