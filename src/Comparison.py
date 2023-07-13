import dtw
import torch
from typing import List, Tuple


class Comparison:

    def __init__(self, word_a: str, hidden_states_a, word_b: str, hidden_states_b, different_at):
        self.word_a = word_a
        self.hidden_states_a = hidden_states_a
        self.word_b = word_b
        self.hidden_states_b = hidden_states_b
        self.diff_at = different_at

        dtw_alignment = self.calculate_alignment()
        self.alignment = list(zip(dtw_alignment.index1, dtw_alignment.index2))

        self.all_path_similarities = None
        self._get_all_path_similarities()

    def get_word_a(self) -> str:
        return self.word_a

    def get_word_b(self) -> str:
        return self.word_b

    def get_different_at(self) -> int:
        return self.diff_at

    def calculate_alignment(self) -> List[Tuple[int, int]]:
        return dtw.dtw(
            # We need cost here, so we subtract the similarities from 1
            x=1-self._calculate_all_similarities(0),
            keep_internals=True,
        )

    def _calculate_all_similarities(self, hidden_layer_index) -> torch.Tensor:
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

    def _get_all_path_similarities(self) -> List[torch.Tensor]:
        if self.all_path_similarities is not None:
            return self.all_path_similarities

        similarities = []
        for hidden_layer_index in range(len(self.hidden_states_a)):
            similarities.append(self.calculate_similarity_along_path(hidden_layer_index))

        self.all_path_similarities = similarities
        return similarities

    def calculate_similarity_along_path(self, hidden_layer_index: int) -> torch.Tensor:
        if self.all_path_similarities is not None:
            return self.all_path_similarities[hidden_layer_index]

        return torch.Tensor([
            torch.cosine_similarity(
                self.hidden_states_a[hidden_layer_index][i],
                self.hidden_states_b[hidden_layer_index][j],
                dim=0,
            )
            for i, j in self.alignment
        ])

    def get_minimum_similarity_along_path(self, hidden_layer_index: int) -> torch.Tensor:
        return torch.min(self.calculate_similarity_along_path(hidden_layer_index))

    def get_relative_position_of_minimum_similarity_along_path(self, hidden_layer_index: int) -> float:
        path_similarities = self.calculate_similarity_along_path(hidden_layer_index)
        index = torch.argmin(path_similarities)
        return index / len(path_similarities)

    def get_std_of_path_similarities(self, hidden_layer_index: int) -> torch.Tensor:
        path_similarities = self.calculate_similarity_along_path(hidden_layer_index)
        return torch.std(path_similarities)
