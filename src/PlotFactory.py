from src import Comparison
from typing import List
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class PlotFactory:

    def __init__(self):
        self.comparisons: List[Comparison] = []
        self.path_similarities: List[List[torch.Tensor]] = []
        self.longest_path_length = None

    def add_comparison(self, comparison: Comparison):
        self.comparisons.append(comparison)

    def initialize_path_similarities(self):
        self.path_similarities = []
        for comparison in self.comparisons:
            self.path_similarities.append(comparison.get_all_path_similarities())

    def plot_all_layers(self):
        for i in range(13):
            self.plot_layer(i)

    def plot_layer(self, layer: int):
        paths_at_layer = [paths[layer] for paths in self.path_similarities]
        dataframes = [self._normalize_and_prepare_dataframe(path) for path in paths_at_layer]
        all_data = pd.concat(dataframes)
        all_data['x_rounded'] = all_data['x'].apply(lambda x: round(x / 0.05) * 0.05)

        plot = sns.lineplot(data=all_data, x='x_rounded', y='y', alpha=0.3)
        plot.set(xlabel='Time', ylabel='Cosine similarity')
        plot.set(xlim=(0, 1), ylim=(0, 1))
        plot.set(title='Cosine similarity along DTW path for all minimal pairs at layer ' + str(layer))
        plt.show()

    def _find_longest_path_length(self) -> int:
        # We assume that all paths from a comparison have the same length, so we just take the first one
        self.longest_path_length = max([len(paths[0]) for paths in self.path_similarities])
        return self.longest_path_length

    def _normalize_and_prepare_dataframe(self, path: torch.Tensor) -> pd.DataFrame:
        dataframe = pd.DataFrame(path.numpy(), columns=['y'])
        dataframe['x'] = dataframe.index / (path.shape[0] - 1)

        return dataframe



