import torch

from src.Comparison import Comparison
from src.Wav2Vec2Runner import Wav2Vec2Runner
from src.WordMapper import WordMapper
from src.minimal_pairs import find_minimal_pairs
import matplotlib.pyplot as plt
import seaborn as sns

def pad_arrays(arrays):
    max_length = max([len(array) for array in arrays])
    padded_arrays = []
    for array in arrays:
        padded_arrays.append(torch.cat([torch.Tensor(array), torch.ones(max_length - len(array))]))
    return padded_arrays

if __name__ == '__main__':
    mapper = WordMapper()
    mapper.import_words('../words/all_words.csv', redundant_at=4)
    mapper.import_recordings('../recordings/segments')

    mapping = mapper.get_word_recording_mapping(filter_out_redundant=True)
    print(mapping.head())

    runner = Wav2Vec2Runner()
    predictions = runner.run(mapping, '../recordings/segments')

    # Generate minimal pairs
    minimal_pairs = find_minimal_pairs(mapping['Value'].to_list())

    all_similarities = []

    i = 0
    for (word_a, word_b), different_at in minimal_pairs:
        if word_a != 'kip' or word_b != 'kik':
            pass

        if different_at != 2:
            continue

        i = i + 1
        if i == 10:
            break

        recording_index = 0
        _, hidden_states_a = predictions[word_a][recording_index]
        _, hidden_states_b = predictions[word_b][recording_index]

        comparison = Comparison(word_a, hidden_states_a, word_b, hidden_states_b, different_at)

        # Some plotting to evaluate the correctness of the alignment
        alignment = comparison.calculate_alignment()
        alignment.plot(type="alignment")
        plt.show()

        similarities = comparison.calculate_all_similarities(0)
        sns.heatmap(similarities)
        plt.show()

        # Calculate the similarities along the best path for all layers
        similarities_along_path = comparison.get_all_path_similarities()
        all_similarities.append(similarities_along_path)

    for hidden_layer_index in range(len(all_similarities[0])):
        similarities_at_layer = [all_similarities[i][hidden_layer_index] for i in range(len(all_similarities))]
        similarities_at_layer = pad_arrays(similarities_at_layer)
        for similarities in similarities_at_layer:
            sns.lineplot(x=range(len(similarities)), y=similarities, alpha=0.3)
        plt.title('Cosine similarity along DTW path for minimal pairs at layer ' + str(hidden_layer_index))
        plt.show()

    grouped = mapping.groupby('Value')
    print(grouped.groups)
    pass
