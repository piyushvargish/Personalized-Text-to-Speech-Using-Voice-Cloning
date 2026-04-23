import torch
import torchaudio
import torch.nn.functional as F
from speechbrain.pretrained import EncoderClassifier

class VoiceSimilarity:

    def __init__(self):

        self.model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb"
        )

    def load_audio(self, file):

        signal, sr = torchaudio.load(file)

        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)

        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            signal = resampler(signal)

        return signal

    def get_embedding(self, file):

        signal = self.load_audio(file)

        with torch.no_grad():
            emb = self.model.encode_batch(signal)

        return emb.squeeze()

    def similarity(self, file1, file2):

        emb1 = self.get_embedding(file1)
        emb2 = self.get_embedding(file2)

        score = F.cosine_similarity(emb1, emb2, dim=0)

        score = float(score)

        score = (score + 1) / 2

        return score