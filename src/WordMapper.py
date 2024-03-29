import pandas as pd
import os
import re
import numpy as np


class WordMapper:

    def __init__(self):
        self.words: pd.DataFrame | None = None
        self.recording_file_name_pattern = re.compile(r'word_(\d+)\.wav')

    def import_words(self, file_name: str, redundant_at: int = None):
        self.words = pd.read_csv(
            file_name,
            delimiter=',',
            header=0,
            dtype={'Group': np.int32, 'Word': np.int32, 'Value': str},
            keep_default_na=False,
            na_values=['_'],
        )
        self.words['is_redundant'] = self.words['Word'] == redundant_at

    def import_recordings(self, directory: str):
        files = os.listdir(directory)
        files = list(filter(self._is_word_recording, files))
        files = sorted(files, key=self._get_recording_index)
        if len(files) > len(self.words):
            raise Exception('More files than words')

        self.words['recording'] = pd.Series(files)
        pass

    def get_word_recording_mapping(self, filter_out_redundant=False) -> pd.DataFrame:
        words = self.words[self.words['recording'].notna()]
        if filter_out_redundant:
            return words[words['is_redundant'] == False]
        return words

    def get_unique_words(self) -> pd.Series:
        return self.get_word_recording_mapping(True)['Value'].unique()

    def _is_word_recording(self, file_name: str) -> bool:
        is_word_recording = re.fullmatch(self.recording_file_name_pattern, file_name) is not None
        if is_word_recording is False:
            print(f'File {file_name} does not match pattern')
        return is_word_recording

    def _get_recording_index(self, file_name: str) -> int:
        return int(re.fullmatch(self.recording_file_name_pattern, file_name).group(1))
