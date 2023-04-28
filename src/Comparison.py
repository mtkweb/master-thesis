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

    def calculate_all_similarities(self, hidden_layer_index):
        similarities_2darray = torch.zeros((
            self.hidden_states_a[hidden_layer_index][0].shape[0],
            self.hidden_states_b[hidden_layer_index][0].shape[0]
        ))
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
            self.hidden_states_a[hidden_layer_index][0],
            self.hidden_states_b[hidden_layer_index][0],
            dist_method='cosine',
            keep_internals=True,
        )
        #return dtw.dtw(
            # calculate distances in advance, maybe with 1-cosine similarity
        #)

    def calculate_distance_along_path(self, alignment: dtw.DTW, hidden_layer_index: int) -> List[torch.Tensor]:
        return [
            torch.cosine_similarity(
                self.hidden_states_a[hidden_layer_index][0][i],
                self.hidden_states_b[hidden_layer_index][0][j],
                dim=0,
            )
            for i, j in zip(alignment.index1, alignment.index2)
        ]
