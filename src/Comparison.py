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
        similarities_2darray = torch.zeros((
            self.hidden_states_a[hidden_layer_index][0].shape[0],
            self.hidden_states_b[hidden_layer_index][0].shape[0]
        ), dtype=torch.double)
        for i in range(self.hidden_states_a[hidden_layer_index][0].shape[0]):
            for j in range(self.hidden_states_b[hidden_layer_index][0].shape[0]):
                similarities_2darray[i][j] = torch.cosine_similarity(
                    self.hidden_states_a[hidden_layer_index][0][i],
                    self.hidden_states_b[hidden_layer_index][0][j],
                    dim=0
                )

        return similarities_2darray

    def get_shortest_path_similarities(self) -> List[List[torch.Tensor]]:
        similarities = []
        for i in range(len(self.hidden_states_a)):
            alignment = self.calculate_alignment(i)
            similarities.append(self.calculate_distance_along_path(alignment, i))
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
                self.hidden_states_a[hidden_layer_index][0][i],
                self.hidden_states_b[hidden_layer_index][0][j],
                dim=0,
            )
            for i, j in zip(alignment.index1, alignment.index2)
        ]
