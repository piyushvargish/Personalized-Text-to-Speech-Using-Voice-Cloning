import streamlit as st
import os
from datetime import datetime
from clone_voice import VoiceCloner
from similarity import VoiceSimilarity
from audiorecorder import audiorecorder

st.set_page_config(page_title="Personalize Text-to-Speech Using Voice Cloning")

st.title("Personalize Text-to-Speech Using Voice Cloning")

samples_dir = "samples"
outputs_dir = "outputs"

os.makedirs(samples_dir, exist_ok=True)
os.makedirs(outputs_dir, exist_ok=True)

sample_path = os.path.join(samples_dir, "sample.wav")

# Default demo text
default_text = "Hello this is a demonstration of an Personalize Text-to-Speech Using Voice Cloning. The generated speech should sound similar to the original speaker"

cloner = VoiceCloner()
similarity_model = VoiceSimilarity()

# ---------- TEXT INPUT ----------
text = st.text_area("Enter text for speech", value=default_text)

# ---------- DEFAULT VOICE SELECTION ----------
st.subheader("Choose Voice Source")

voice_option = st.selectbox(
    "Select Voice",
    ["Default Male", "Default Female", "Upload / Record"]
)

# Paths for default voices
male_voice_path = os.path.join(samples_dir, "male.wav")
female_voice_path = os.path.join(samples_dir, "female.wav")

# ---------- HANDLE DEFAULT VOICES ----------
if voice_option == "Default Male":

    if os.path.exists(male_voice_path):
        sample_path = male_voice_path
        st.success("Using default male voice")
        st.audio(sample_path)
    else:
        st.error("male.wav not found in samples folder")

elif voice_option == "Default Female":

    if os.path.exists(female_voice_path):
        sample_path = female_voice_path
        st.success("Using default female voice")
        st.audio(sample_path)
    else:
        st.error("female.wav not found in samples folder")

# ---------- UPLOAD / RECORD ----------
elif voice_option == "Upload / Record":

    tab1, tab2 = st.tabs(["Upload Voice", "Record Voice"])

    with tab1:
        uploaded_file = st.file_uploader(
            "Upload voice sample (wav)",
            type=["wav"]
        )

        if uploaded_file:
            sample_path = os.path.join(samples_dir, "sample.wav")

            with open(sample_path, "wb") as f:
                f.write(uploaded_file.read())

            st.success("Voice uploaded")
            st.audio(sample_path)

    with tab2:
        st.write("Record your voice")

        audio = audiorecorder(
            "Start Recording",
            "Stop Recording"
        )

        if len(audio) > 0:
            sample_path = os.path.join(samples_dir, "sample.wav")

            audio.export(sample_path, format="wav")

            st.success("Voice recorded")
            st.audio(sample_path)

# ---------- GENERATE VOICE ----------
if st.button("Generate Cloned Voice"):

    if not os.path.exists(sample_path):
        st.error("Provide a voice sample first")

    elif text.strip() == "":
        st.error("Enter text")

    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        output_file = os.path.join(
            outputs_dir,
            f"voice_{timestamp}.wav"
        )

        with st.spinner("Generating voice..."):

            cloner.clone(
                text,
                sample_path,
                output_file
            )

        st.success("Voice generated")
        st.audio(output_file)

        # ---------- SIMILARITY ----------
        score = similarity_model.similarity(
            sample_path,
            output_file
        )

        percent = score * 100

        st.subheader("Voice Similarity")

        st.progress(score)

        st.write(f"Similarity Score: {percent:.2f}%")

        # ---------- DOWNLOAD ----------
        with open(output_file, "rb") as f:
            st.download_button(
                "Download Voice",
                f,
                file_name="cloned_voice.wav"
            )