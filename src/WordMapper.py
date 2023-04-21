import pandas as pd
import os
import re


class WordMapper:

    def __init__(self):
        self.words: pd.DataFrame | None = None
        self.recording_file_name_pattern = re.compile(r'word_(\d+)\.wav')

    def import_words(self, file_name: str, redundant_at: int = None):
        self.words = pd.read_csv(file_name)
        self.words['is_redundant'] = self.words['Word'] == redundant_at

    def import_recordings(self, directory: str):
        files = os.listdir(directory)
        if len(files) > len(self.words):
            raise Exception('More files than words')
        files = list(filter(self._is_word_recording, files))
        files = sorted(files, key=self._get_recording_index)

        self.words['recording'] = pd.Series(files)
        pass

    def get_word_recording_mapping(self) -> pd.DataFrame:
        return self.words[self.words['recording'].notna()]

    def _is_word_recording(self, file_name: str) -> bool:
        return re.fullmatch(self.recording_file_name_pattern, file_name) is not None

    def _get_recording_index(self, file_name: str) -> int:
        return int(re.fullmatch(self.recording_file_name_pattern, file_name).group(1))
