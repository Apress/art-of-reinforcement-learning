from typing import Mapping, Text, List
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def load_experiments(experiments: List[Mapping[Text, str]]) -> pd.DataFrame:
    """
    Expect experiments to be a list of dicts.

    For example:
    ```
    experiments = [
        dict(
            agent_id='dqn',
            env_name='pong',
            base_path='./logs/dqn/pong/',
        ),
    ]
    ```

    """

    df_list = []
    for experiment in experiments:
        for seed in range(1, 11):
            csv_file = os.path.join(experiment['base_path'], f'{seed}/results.csv')
            if os.path.exists(csv_file):
                with open(csv_file, 'r') as f:
                    df = pd.read_csv(f)
                    df = df.assign(
                        agent_id=experiment['agent_id'],
                        environment_name=experiment['env_name'],
                        seed=seed,
                    )
                    df_list.append(df)

    df_raw = pd.concat(df_list, sort=True).reset_index(drop=True)
    return df_raw


def moving_average(values, window_size):
    """Smooth data using moving average with a window size."""
    # numpy.convolve uses zero for initial missing values, so is not suitable.
    numerator = np.nancumsum(values)
    # The sum of the last window_size values.
    numerator[window_size:] = numerator[window_size:] - numerator[:-window_size]
    denominator = np.ones(len(values)) * window_size
    denominator[:window_size] = np.arange(1, window_size + 1)
    smoothed = numerator / denominator
    assert values.shape == smoothed.shape
    return smoothed


def smooth_fn(df, smoothing_window, index_columns, columns):
    dfg = df.groupby(index_columns)
    for col in columns:
        df[col] = dfg[col].transform(lambda s: moving_average(s.values, smoothing_window))
    return df


def group_by_and_compute_ci_interval(df, index_columns, columns, ci=0.95):
    """Group by columns then computes 95% confidence interval"""

    assert ci > 0

    dfg = df.groupby(index_columns)
    agg_df = dfg.mean()

    for col in columns:
        mean = dfg[col].mean()
        std = dfg[col].std()
        z_score = 1.01 + ci  # 1.96 for 95% confidence level, two-tailed
        n = len(dfg[col])
        std_err = std / np.sqrt(n)
        error = z_score * std_err
        agg_df[f'{col}_ci_low'] = mean - error
        agg_df[f'{col}_ci_high'] = mean + error

    agg_df = agg_df.reset_index()

    return agg_df


def load_and_process_experiments(
    experiments,
    columns,
    smoothing_index_columns=['agent_id', 'environment_name', 'seed'],
    ci_index_columns=['agent_id', 'environment_name', 'step'],
    smoothing_window=5,
    ci=0.95,
):
    # Load csv files
    df = load_experiments(experiments)

    # Smooth data using a window size of N
    if smoothing_window > 0:
        df = df.pipe(
            smooth_fn,
            smoothing_window=smoothing_window,
            columns=columns,
            index_columns=smoothing_index_columns,
        )

    df = group_by_and_compute_ci_interval(df, ci_index_columns, columns, ci)

    return df


def plot_and_save_experiments(
    experiments,
    save_path='',
    show_train_data=True,
    show_eval_data=True,
    additional_columns=[],
):
    columns = []
    if show_train_data:
        columns.append('train_episode_return')
    if show_eval_data:
        columns.append('eval_episode_return')

    if additional_columns:
        columns.extend(additional_columns)

    df = load_and_process_experiments(experiments, columns)

    agent_ids = df['agent_id'].unique()
    environment_names = df['environment_name'].unique()

    # Loop over the data and plot each line with a different built-in color
    colors = list(mcolors.BASE_COLORS.values())

    i = 0
    for agent in agent_ids:
        for col in columns:
            plt.plot(
                'step',
                col,
                data=df[df['agent_id'] == agent],
                color=colors[i],
                label=f'{agent}-{col}',
            )

            plt.fill_between(
                x='step',
                y1=f'{col}_ci_high',
                y2=f'{col}_ci_low',
                data=df[df['agent_id'] == agent],
                color=colors[i],
                alpha=0.5,
            )

            i += 1

    plt.legend()
    plt.ylabel('Episode return')
    plt.xlabel('Steps')
    plt.show()

    if save_path is None or save_path == '' or not os.path.exists(save_path):
        return

    # Save to csv file
    header = ['agent_id', 'environment_name', 'step']

    for col in columns:
        header.append(col)
        header.append(f'{col}_ci_low')
        header.append(f'{col}_ci_high')

    for env_name in environment_names:
        for agent in agent_ids:
            full_path = os.path.join(save_path, f'{agent}_{env_name}.csv')
            df.loc[(df['agent_id'] == agent) & (df['environment_name'] == env_name)].to_csv(
                full_path, columns=header, index=False
            )
            print(f'Csv file saved to {full_path}')
