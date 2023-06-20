from src import Comparison
from typing import List
import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class PlotFactory:

    def __init__(self, save_figures: bool, speaker: str):
        self.comparisons: List[Comparison] = []
        self.path_similarities: List[List[torch.Tensor]] = []
        self.longest_path_length = None

        self.save_figures = save_figures
        self.speaker = speaker

    def add_comparison(self, comparison: Comparison):
        self.comparisons.append(comparison)

    def plot_all_layers(self):
        for i in range(13):
            self.plot_layer(i)

    def plot_layer(self, layer: int):
        dataframes = [
            self._normalize_and_prepare_dataframe(
                comparison.calculate_similarity_along_path(layer),
                comparison.get_different_at(),
            )
            for comparison in self.comparisons
        ]
        all_data = pd.concat(dataframes)
        all_data['x_rounded'] = all_data['x'].apply(lambda x: round(x / 0.05) * 0.05)

        plot = sns.lineplot(data=all_data, x='x_rounded', y='y', hue='different_at', alpha=0.3, errorbar=('pi', 50))
        plot.set(xlabel='Time', ylabel='Cosine similarity')
        plot.set(xlim=(0, 1), ylim=(0, 1))
        plot.set(title='Cosine similarity along DTW path for all minimal pairs at layer ' + str(layer) + '\nfor the ' + self.speaker + ' speaker')

        if self.save_figures:
            plt.savefig('layer_' + str(layer) + '.png')
        plt.show()

    def _find_longest_path_length(self) -> int:
        # We assume that all paths from a comparison have the same length, so we just take the first one
        self.longest_path_length = max([len(paths[0]) for paths in self.path_similarities])
        return self.longest_path_length

    def _normalize_and_prepare_dataframe(self, path: torch.Tensor, different_at: int) -> pd.DataFrame:
        dataframe = pd.DataFrame(path.numpy(), columns=['y'])
        dataframe['x'] = dataframe.index / (path.shape[0] - 1)
        dataframe['different_at'] = different_at

        return dataframe
