import os
from pydub import AudioSegment
from pydub.silence import split_on_silence


class Splitter:

    def __init__(self,
                 start_index: int = 0,
                 min_silence_len: int = 1000,
                 silence_thresh: int = -16,
                 keep_silence: int = 100,
                 seek_step: int = 1,
                 ):
        self.start_index = start_index
        self.min_silence_len = min_silence_len
        self.silence_thresh = silence_thresh
        self.keep_silence = keep_silence
        self.seek_step = seek_step

    def split(self, file: str, output_dir: str):
        sound_file = AudioSegment.from_wav(file)
        audio_chunks = split_on_silence(sound_file, self.min_silence_len, self.silence_thresh, self.keep_silence,
                                        self.seek_step)
        for i, chunk in enumerate(audio_chunks):
            out_file = os.path.join(output_dir, f"word_{i + self.start_index}.wav")
            print("exporting", out_file)
            chunk.export(out_file, format="wav")

        return audio_chunks
