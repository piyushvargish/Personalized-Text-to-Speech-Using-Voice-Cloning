from TTS.api import TTS
import torch
from audio_processor import process_audio

class VoiceCloner:

    def __init__(self):

        device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tts = TTS(
            model_name="tts_models/multilingual/multi-dataset/your_tts",
            progress_bar=False
        ).to(device)

    def clone(self, text, speaker_wav, output_file):

        processed_audio = "samples/processed.wav"

        process_audio(
            speaker_wav,
            processed_audio
        )

        # Natural speech parameters
        self.tts.tts_to_file(
    text=text,
    speaker_wav=processed_audio,
    language="en",
    temperature=0.7,
    speed=0.98,
    length_penalty=1.0,
    repetition_penalty=2.5,
    top_p=0.8,
    file_path=output_file
)
            
        