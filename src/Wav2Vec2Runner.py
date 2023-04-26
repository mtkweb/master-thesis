import pandas as pd
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2Config
import librosa
import numpy as np
import torch
from typing import Tuple, List
import os


class Wav2Vec2Runner:
    MODEL_NAME = 'facebook/wav2vec2-base-960h'
    SAMPLE_RATE = 16000

    def __init__(self):
        self._initialize_model()

    def _initialize_model(self):
        self.model = Wav2Vec2ForCTC.from_pretrained(Wav2Vec2Runner.MODEL_NAME)
        self.processor = Wav2Vec2Processor.from_pretrained(Wav2Vec2Runner.MODEL_NAME)
        self.config = Wav2Vec2Config.from_pretrained(Wav2Vec2Runner.MODEL_NAME)

    def run(self, mapping: pd.DataFrame, directory: str) -> List[Tuple[List[str], Tuple[torch.Tensor]]]:
        dataset = [self._load_recording(os.path.join(directory, file_name)) for file_name in mapping['recording']]
        dataset = [self._process_recording(recording) for recording in dataset]

        dataset_dict = {}
        for word, (transcription, hidden_states) in zip(mapping['Value'], dataset):
            if word not in dataset_dict:
                dataset_dict[word] = []
            dataset_dict[word].append((transcription, hidden_states))

        return dataset_dict

    def _load_recording(self, file_name: str) -> np.ndarray:
        speech_array, sampling_rate = librosa.load(file_name, sr=Wav2Vec2Runner.SAMPLE_RATE)
        return speech_array

    def _process_recording(self, recording: np.ndarray) -> Tuple[List[str], Tuple[torch.Tensor]]:
        input_values = self.processor(
            recording,
            sampling_rate=Wav2Vec2Runner.SAMPLE_RATE,
            return_tensors="pt",
            padding='longest',
        ).input_values

        with torch.no_grad():
            model_output = self.model(input_values, output_hidden_states=True)
            logits = model_output.logits

        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)
        hidden_states = model_output.hidden_states

        return transcription, hidden_states
