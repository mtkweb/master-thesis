from src.Wav2Vec2Runner import Wav2Vec2Runner
from src.WordMapper import WordMapper
from src.minimal_pairs import find_minimal_pairs

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

    grouped = mapping.groupby('Value')
    print(grouped.groups)
    pass
