from src.Comparison import Comparison
from src.PlotFactory import PlotFactory
from src.Wav2Vec2Runner import Wav2Vec2Runner
from src.WordMapper import WordMapper
from src.minimal_pairs import find_minimal_pairs
from src.calculate_ranking import calculate_ranking
import matplotlib.pyplot as plt
import seaborn as sns


if __name__ == '__main__':
    mapper = WordMapper()
    mapper.import_words('../words/all_words.csv', redundant_at=4)
    mapper.import_recordings('../recordings/segments')

    mapping = mapper.get_word_recording_mapping(filter_out_redundant=True)
    mapping = calculate_ranking(mapping, mapper.get_unique_words())
    mapping = mapping[mapping['rank'] == 1.0]
    print(mapping.head())

    grouped = mapping.groupby('Value')
    print(grouped.groups)

    # Generate minimal pairs
    minimal_pairs = find_minimal_pairs(mapper.get_unique_words().tolist())

    runner = Wav2Vec2Runner(use_carrier_phrase=False)
    predictions = runner.run(mapping, '../recordings/segments')

    plot_factory = PlotFactory()
    i = 0
    for (word_a, word_b), different_at in minimal_pairs:
        if word_a != 'kip' or word_b != 'kik':
            pass

        if different_at != 2:
            pass

        i = i + 1
        if i == 100:
            pass

        recording_index = 0
        _, hidden_states_a = predictions[word_a][recording_index]
        _, hidden_states_b = predictions[word_b][recording_index]

        comparison = Comparison(word_a, hidden_states_a, word_b, hidden_states_b, different_at)

        # Some plotting to evaluate the correctness of the alignment
        #alignment = comparison.calculate_alignment()
        #alignment.plot(type="alignment")
        #plt.show()

        #similarities = comparison.calculate_all_similarities(0)
        #sns.heatmap(similarities)
        #plt.show()

        # Calculate the similarities along the best path for all layers
        #similarities_along_path = comparison.get_all_path_similarities()
        #all_similarities.append(similarities_along_path)
        plot_factory.add_comparison(comparison)

    plot_factory.plot_all_layers()
