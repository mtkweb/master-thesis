import glob
import pandas as pd
import os
from typing import List


def calculate_grouped_table(dataframe: pd.DataFrame, column: List[str], label: List[str]) -> pd.DataFrame:
    dataframe = dataframe[['layer', 'different_at', *column]]
    dataframe.columns = ['Layer', 'Different at', *label]
    grouped_df = dataframe.groupby(['Layer', 'Different at']).agg(['mean', 'std'])
    grouped_df = grouped_df.reset_index()

    grouped_df = grouped_df.pivot(index='Layer', columns='Different at')
    grouped_df = grouped_df.style.format('{:.2f}')
    print(grouped_df.to_latex(
        environment='table',
        clines='skip-last;index',
        position_float='centering',
        multicol_align='c',
        hrules=True,
    ))

if __name__ == '__main__':
    data_files = glob.glob(os.path.join('analysis_data', '*.pkl'))
    print(data_files)
    dataframes = [(os.path.basename(file).split('.')[0], pd.read_pickle(file)) for file in data_files]

    grouped_df = calculate_grouped_table(
        dataframes[2][1],
        ['position_of_minimum', 'minimum_similarity', 'similarity_std'],
        ['Timestamp of minimum', 'Minimum similarity', 'Standard deviation of similarity']
        #['position_of_minimum'],
        #['Timestamp of minimum']
    )

