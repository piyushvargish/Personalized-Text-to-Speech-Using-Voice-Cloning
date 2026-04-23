# Personalized Text-to-Speech Using Voice Cloning

## 📌 Overview
This project implements a personalized text-to-speech (TTS) system using voice cloning. It can generate speech in a target speaker’s voice using a small audio sample.

## 🚀 Features
- Voice cloning from sample audio
- Text-to-speech generation
- Speaker similarity estimation
- Clean and modular Python implementation

## 🛠️ Tech Stack
- Python
- Deep Learning (PyTorch)
- Audio Processing

## 📂 Project Structure
- `app.py` → Main application
- `clone_voice.py` → Voice cloning logic
- `audio_processor.py` → Audio preprocessing
- `similarity.py` → Similarity calculation
- `samples/` → Sample inputs/outputs
- `pretrained_models/` → Model references

## ⚙️ Installation
```bash
pip install -r requirements.txt
▶️ Usage
python app.py
📊 Results
Achieved ~60–70% voice similarity
Performance depends on dataset quality and model tuning
⚠️ Note

Pretrained models are not included due to size limitations.

🔮 Future Improvements
Improve voice accuracy
Real-time speech generation
Emotion control
Web interface
👨‍💻 Author

Piyush Vargish
