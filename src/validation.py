from src.Comparison import Comparison
from src.PlotFactory import PlotFactory
from src.Wav2Vec2Runner import Wav2Vec2Runner
from src.WordMapper import WordMapper
from src.calculate_ranking import calculate_ranking


if __name__ == '__main__':
    SEGMENTS_DIRECTORY = '../recordings/segments-male'
    CARRIER_PHRASE_PATH = '../recordings/carrier_phrase_16k_male.wav'

    mapper = WordMapper()
    mapper.import_words('../words/all_words.csv', redundant_at=4)
    mapper.import_recordings(SEGMENTS_DIRECTORY)

    mapping = mapper.get_word_recording_mapping(filter_out_redundant=True)
    mapping = calculate_ranking(mapping, mapper.get_unique_words(), SEGMENTS_DIRECTORY)
    mapping = mapping[mapping['rank'] <= 2.0]
    print(mapping.head())

    unique_words = mapper.get_unique_words()

    runner = Wav2Vec2Runner(use_carrier_phrase=False, carrier_phrase_path=CARRIER_PHRASE_PATH)
    predictions = runner.run(mapping, SEGMENTS_DIRECTORY)

    plot_factory = PlotFactory(save_figures=False, speaker='male')
    for word in unique_words:
        _, hidden_states_a = predictions[word][0]
        _, hidden_states_b = predictions[word][1]

        comparison = Comparison(word, hidden_states_a, word, hidden_states_b, 0)
        plot_factory.add_comparison(comparison)

    plot_factory.plot_all_layers()
