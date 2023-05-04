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
    # For each minimal pair, apply DTW to the hidden states
    #hidden_layer_index = 7


    for hidden_layer_index in range(13):
        distance_paths = []
        i = 0
        for (word_a, word_b), different_at in minimal_pairs:
            if word_a != 'kip' or word_b != 'kik':
                continue

            if different_at != 2:
                continue

            i = i + 1
            if i == 10:
                break

            recording_index = 0
            _, hidden_states_a = predictions[word_a][recording_index]
            _, hidden_states_b = predictions[word_b][recording_index]

            comparison = Comparison(word_a, hidden_states_a, word_b, hidden_states_b, different_at)

            alignment = comparison.calculate_alignment(hidden_layer_index)
            alignment.plot(type="alignment")
            plt.show()

            similarities = comparison.calculate_all_similarities(hidden_layer_index)
            sns.heatmap(similarities)
            plt.show()

            distance_along_path = comparison.calculate_distance_along_path(alignment, hidden_layer_index)
            distance_paths.append(distance_along_path)

        distance_paths = pad_arrays(distance_paths)

        for i in range(0, len(distance_paths)):
            sns.lineplot(x=range(len(distance_paths[0])), y=distance_paths[i], alpha=0.3)
        plt.title('Cosine similarity (inverted) along DTW path for minimal pairs at layer ' + str(hidden_layer_index))
        #plt.savefig('cosine_similarity_along_dtw_path_layer' + str(hidden_layer_index) + '.png')
        plt.show()

    grouped = mapping.groupby('Value')
    print(grouped.groups)
    pass
