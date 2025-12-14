# Voice Attractiveness Prediction

This project predicts perceived voice attractiveness (Mean Opinion Score – MOS)
from speech signals using machine learning and deep learning models.

## Project Overview
The system extracts acoustic features from audio recordings and predicts
a continuous attractiveness score. Two modeling approaches are used:
- Support Vector Regression (SVR) on handcrafted audio features
- CNN + BiLSTM neural network on mel-spectrograms

Additionally, Grad-CAM is applied to visualize which time–frequency regions
of the spectrogram contribute most to the model’s predictions.

## Project Structure
```
projektAnjaGH.py # main training and evaluation script
demo_app.py # interactive Gradio demo
requirements.txt # Python dependencies
README.md
.gitignore
```

## Dataset
The dataset used in this project is **not included** in the repository due to
licensing and size constraints.

Audio files should be placed in:
```
data/wav/
```

## Installation
```bash
pip install -r requirements.txt
```

## Running the Project

Train and evaluate the models:
```bash
python projektAnjaGH.py
```

Run the interactive demo:
```bash
python demo_app.py
```

## Notes
- Trained models and extracted features are not uploaded to GitHub.
- Paths are handled using relative directories for portability.

## Author
Anja Bulajić