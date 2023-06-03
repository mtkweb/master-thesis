from src.WordMapper import WordMapper
from pydub import AudioSegment
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def calculate_duration(file: str) -> int:
    return len(AudioSegment.from_wav(file))


def get_ranking_within_group(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values('duration')
    group['rank'] = group['duration'].rank(method='dense')
    return group


if __name__ == '__main__':
    mapper = WordMapper()
    mapper.import_words('../words/all_words.csv', redundant_at=4)
    mapper.import_recordings('../recordings/segments')

    mapping = mapper.get_word_recording_mapping(filter_out_redundant=True)
    print(mapping.head())

    mapping['duration'] = mapping['recording'].apply(lambda file: calculate_duration(f'../recordings/segments/{file}'))

    grouped = mapping.groupby('Value')
    print(grouped.groups)

    group_data = {}
    durations_df = None
    unique_words = mapper.get_unique_words()
    for unique_word in unique_words:
        group_df = grouped.get_group(unique_word)
        group_df = group_df.iloc[0:4]

        group_data[unique_word] = {
            'word': unique_word,
            'average_duration': group_df['duration'].mean(),
            'min_duration': group_df['duration'].min(),
            'max_duration': group_df['duration'].max(),
            'std_duration': group_df['duration'].std(),
            'relative_std_duration': group_df['duration'].std() / group_df['duration'].mean(),
        }

        group_df['diff_to_average'] = group_df['duration'] - group_data[unique_word]['average_duration']
        group_df['rank'] = group_df['diff_to_average'].abs().rank(method='first')

        if durations_df is None:
            durations_df = group_df
        else:
            durations_df = pd.concat([durations_df, group_df])

    mapping = mapping.merge(
        durations_df[['diff_to_average', 'rank']],
        how='left',
        left_index=True,
        right_index=True,
        suffixes=(None, None),
    )

    group_data_df = pd.DataFrame.from_dict(group_data, orient='index')
    print(group_data_df.head())

    sns.boxplot(data=group_data_df, x='relative_std_duration')
    plt.show()

    """
        - 1. Calculate average duration in a group
        - 2. Remove the 2 most deviating recordings
        - 3. Compare the similarity of the remaining recordings
        
        Calculate difference with average duration for each recording
        Rank the recordings within the group from 1 to 4
        Remove recordings with rank 3 and 4
        Compare the similarity of the recordings with rank 1 and 2
    """
