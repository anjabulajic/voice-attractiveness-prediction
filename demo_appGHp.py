"""
Gradio demo application for voice attractiveness prediction
using a trained CNN-BiLSTM model.
"""

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
import numpy as np
import librosa
import tensorflow as tf
import gradio as gr

#Paths
RESULTS_PATH = os.path.join(BASE_DIR, "results")
MODEL_PATH = os.path.join(RESULTS_PATH, "cnn_lstm_mos_model.keras")
REF_STATS_PATH = os.path.join(RESULTS_PATH, "reference_mos_stats.npz")

#Loading the Model and Reference MOS Scores
model = tf.keras.models.load_model(MODEL_PATH)
ref_data = np.load(REF_STATS_PATH)
ref_mos = ref_data["all_mos"]
ref_mean = float(ref_data["mean_mos"])
ref_std = float(ref_data["std_mos"])


#Same Preprocessing as in the Project
def audio_to_mel_demo(file_path, max_len=128):
    y, sr = librosa.load(file_path, sr=16000)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    #Per-sample z-score normalization
    m = np.mean(mel_db)
    s = np.std(mel_db) + 1e-8
    mel_db = (mel_db - m) / s

    #Fixed length of 128 frames
    if mel_db.shape[1] < max_len:
        pad = max_len - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad)), mode="constant")
    else:
        mel_db = mel_db[:, :max_len]

    mel_db = mel_db[..., np.newaxis]  # (64, 128, 1)
    return mel_db


def predict_mos(audio):
        if audio is None:
            return "Please upload or record a voice."

        if isinstance(audio, str):
            file_path = audio
        else:
            sr, y = audio
            file_path = "temp_input.wav"
            librosa.output.write_wav(file_path, y, sr)

        mel = audio_to_mel_demo(file_path)
        mel = np.expand_dims(mel, axis=0)  

        pred = model.predict(mel, verbose=0)[0, 0]
        mos = float(pred)

        percentile = (ref_mos < mos).mean() * 100.0

        text = (
            f"Predicted level of attractiveness (MOS): **{mos:.2f}**\n\n"
            f"Average MOS in the dataset: {ref_mean:.2f}\n"
            f"\nYour voice is higher than approximately {percentile:.1f}% of voices in the dataset."
        )

        if mos >= ref_mean + 0.5 * ref_std:
            text += "\n\n Your voice is above average in terms of attractiveness rating."
        elif mos <= ref_mean - 0.5 * ref_std:
            text += "\n\n Your voice is below average in this dataset."
        else:
            text += "\n\n Your voice is close to the average in this dataset."

        return text


#Gradio Interface
demo = gr.Interface(
    fn=predict_mos,
    inputs=gr.Audio(
    sources=["microphone", "upload"],
    type="filepath",
    label="Record or upload your voice."
),
    outputs=gr.Markdown(label="Rezultat"),
    title="Voice Attractiveness Demo",
    description=(
        "Record a short speech sample. The model will estimate your voice attractiveness (MOS) and show how your voice compares to those in the CocoNut-Humoresque dataset."
    ),
)

if __name__ == "__main__":
    demo.launch()
