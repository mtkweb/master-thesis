import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import glob
from typing import List, Tuple


def correlation_diffat_vs_position_of_minimum(dataframes: List[Tuple[str, pd.DataFrame]]):
    correlation_dataframes = []
    for name, df in dataframes:
        correlation_values = []
        for i in range(df['layer'].max() + 1):
            df_filtered = df[df['layer'] == i]
            correlation = df_filtered['different_at'].corr(df_filtered['position_of_minimum'])
            correlation_values.append((i, correlation))

        correlation_df = pd.DataFrame(correlation_values, columns=['layer', 'correlation'])
        correlation_df['series'] = name
        correlation_dataframes.append(correlation_df)

    sns.lineplot(data=pd.concat(correlation_dataframes), hue='series', x='layer', y='correlation')
    plt.title('Correlation between position of minimum and different_at per layer')
    plt.ylim(0, 1)
    plt.xticks(range(13))
    plt.show()

def layer_vs_similarity_std(dataframes: List[Tuple[str, pd.DataFrame]], replacement_value):
    correlation_dataframes = []
    for name, df in dataframes:
        df['different_at'] = df['different_at'].apply(lambda x: replacement_value if x == 0 else x)
        correlation_values = []
        for i in range(df['layer'].max() + 1):
            df_filtered = df[df['layer'] == i]
            correlation = df_filtered['different_at'].corr(df_filtered['similarity_std'])
            correlation_values.append((i, correlation))

        correlation_df = pd.DataFrame(correlation_values, columns=['layer', 'correlation'])
        correlation_df['series'] = name
        correlation_dataframes.append(correlation_df)

    sns.lineplot(data=pd.concat(correlation_dataframes), hue='series', x='layer', y='correlation')
    plt.title('Correlation between the consonant being different\nand the standard deviation of the similarity')
    plt.ylim(0, 1)
    plt.xticks(range(13))
    plt.show()

def plot_histograms(comparison_data: pd.DataFrame):
    x_variables = [
        ('position_of_minimum', 0.05),
        ('minimum_similarity', 0.05),
        ('similarity_std', 0.02),
    ]
    for variable, binwidth in x_variables:
        g = sns.FacetGrid(comparison_data, row='layer', col='different_at')
        g.map_dataframe(sns.histplot, x=variable,  stat='probability', binwidth=binwidth)
        plt.show()



if __name__ == '__main__':
    data_files = glob.glob(os.path.join('analysis_data', '*.pkl'))
    dataframes = [(os.path.basename(file).split('.')[0], pd.read_pickle(file)) for file in data_files]

    #all_data_df = pd.concat([df for name, df in dataframes], keys=[name for name, df in dataframes], names=['name', 'index'])

    #correlation_diffat_vs_position_of_minimum(dataframes)
    layer_vs_similarity_std(dataframes, 1.5)
