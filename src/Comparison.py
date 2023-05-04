import dtw
import torch
from typing import List


class Comparison:

    def __init__(self, word_a: str, hidden_states_a, word_b: str, hidden_states_b, different_at):
        self.word_a = word_a
        self.hidden_states_a = hidden_states_a
        self.word_b = word_b
        self.hidden_states_b = hidden_states_b
        self.diff_at = different_at

    def calculate_all_similarities(self, hidden_layer_index) -> torch.Tensor:
        hidden_states_a_at_layer = self.hidden_states_a[hidden_layer_index]
        hidden_states_b_at_layer = self.hidden_states_b[hidden_layer_index]

        duration_a = hidden_states_a_at_layer.shape[0]
        duration_b = hidden_states_b_at_layer.shape[0]

        similarities_2d_array = torch.zeros((duration_a, duration_b), dtype=torch.double)
        for i in range(duration_a):
            for j in range(duration_b):
                similarities_2d_array[i][j] = torch.cosine_similarity(
                    hidden_states_a_at_layer[i],
                    hidden_states_b_at_layer[j],
                    dim=0
                )

        return similarities_2d_array

    def get_shortest_path_similarities(self) -> List[List[torch.Tensor]]:
        similarities = []
        for hidden_layer_index in range(len(self.hidden_states_a)):
            alignment = self.calculate_alignment(hidden_layer_index)
            similarities.append(self.calculate_distance_along_path(alignment, hidden_layer_index))
        return similarities

    def calculate_alignment(self, hidden_layer_index: int) -> dtw.DTW:
        return dtw.dtw(
            # We need cost here, so we subtract the similarities from 1
            x=1-self.calculate_all_similarities(hidden_layer_index),
            keep_internals=True,
        )

    def calculate_distance_along_path(self, alignment: dtw.DTW, hidden_layer_index: int) -> List[torch.Tensor]:
        return [
            torch.cosine_similarity(
                self.hidden_states_a[hidden_layer_index][i],
                self.hidden_states_b[hidden_layer_index][j],
                dim=0,
            )
            for i, j in zip(alignment.index1, alignment.index2)
        ]
