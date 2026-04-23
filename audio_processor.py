import librosa
import soundfile as sf
import noisereduce as nr
import numpy as np

def process_audio(
    input_path,
    output_path,
    target_sr=22050,
    min_duration=5,
    trim_db=20,
    noise_reduce=True
):
    """
    Clean and prepare audio for voice cloning or ML models.
    """

    try:
        # ---------- Load Audio ----------
        audio, sr = librosa.load(input_path, sr=None)

        # ---------- Resample if Needed ----------
        if sr != target_sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            sr = target_sr

        # ---------- Noise Reduction ----------
        if noise_reduce:
            audio = nr.reduce_noise(y=audio, sr=sr)

        # ---------- Trim Silence ----------
        audio, _ = librosa.effects.trim(audio, top_db=trim_db)

        # ---------- Normalize Volume (RMS) ----------
        rms = np.sqrt(np.mean(audio**2))
        if rms > 0:
            target_rms = 0.1
            audio = audio * (target_rms / rms)

        # ---------- Peak Normalization ----------
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val

        # ---------- Ensure Minimum Duration ----------
        min_len = int(sr * min_duration)
        if len(audio) < min_len:
            pad_length = min_len - len(audio)
            audio = np.pad(audio, (0, pad_length), mode="constant")

        # ---------- Fade In/Out (Avoid Clicks) ----------
        fade_samples = int(0.02 * sr)  # 20ms fade
        fade_in = np.linspace(0, 1, fade_samples)
        fade_out = np.linspace(1, 0, fade_samples)

        audio[:fade_samples] *= fade_in
        audio[-fade_samples:] *= fade_out

        # ---------- Remove DC Offset ----------
        audio = audio - np.mean(audio)

        # ---------- Save Audio ----------
        sf.write(output_path, audio, sr)

        return True, "Audio processed successfully"

    except Exception as e:
        return False, str(e)