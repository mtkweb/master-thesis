from pydub import AudioSegment
import pandas as pd


def calculate_duration(file: str) -> int:
    return len(AudioSegment.from_wav(file))


def get_ranking_within_group(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values('duration')
    group['rank'] = group['duration'].rank(method='dense')
    return group


def calculate_ranking(mapping: pd.DataFrame, unique_words: pd.Series) -> pd.DataFrame:
    mapping['duration'] = mapping['recording'].apply(lambda file: calculate_duration(f'../recordings/segments/{file}'))
    grouped = mapping.groupby('Value')

    group_data = {}
    durations_df = None
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

    return mapping.merge(
        durations_df[['diff_to_average', 'rank']],
        how='left',
        left_index=True,
        right_index=True,
        suffixes=(None, None),
    )
