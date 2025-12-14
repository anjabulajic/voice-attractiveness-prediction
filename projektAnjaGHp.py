"""
Voice Attractiveness Prediction Project
Main script for data processing, feature extraction,
model training, evaluation and Grad-CAM analysis.
"""

import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.optimizers import AdamW

plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12
})


#Data Paths

DATA_PATH = os.path.join(BASE_DIR, "data")
AUDIO_PATH = os.path.join(DATA_PATH, "wav")
RESULTS_PATH = os.path.join(BASE_DIR, "results")

os.makedirs(RESULTS_PATH, exist_ok=True)


#Loading Attribute CSV Files (Speaker Gender)

ATTRIBUTE_PATH = os.path.join(DATA_PATH, "atribute")

def infer_gender_row(row):
    #Based on the gender_1–gender_5 columns, the system determines whether the speaker’s voice is male or female.
    #It checks Japanese characters, where '女' corresponds to female (F) and '男' corresponds to male (M).
    
    vals = []
    for c in row.index:
        if str(c).startswith("gender_"):
            v = row[c]
            if isinstance(v, str) and v.strip():
                vals.append(v)

    #'女' - female
    for v in vals:
        if "女" in v:
            return "F"
    #'男' - male
    for v in vals:
        if "男" in v:
            return "M"
    return None

gender_attr_dfs = []
for split in ["train", "valid", "test"]:
    attr_file = os.path.join(ATTRIBUTE_PATH, f"{split}.csv")
    attr_df = pd.read_csv(attr_file)
    attr_df["id"] = attr_df["id"].astype(int)
    g = attr_df.apply(infer_gender_row, axis=1)
    gender_attr_dfs.append(pd.DataFrame({
        "id": attr_df["id"],
        "gender": g
    }))

#all three attribute files (train/validation/test) are merged into a single dataset
gender_attr_df = pd.concat(gender_attr_dfs, ignore_index=True)

#in the case of duplicate IDs, the first occurrence is kept and the others are discarded
gender_attr_df = gender_attr_df.drop_duplicates(subset=["id"], keep="first")

print("Number of IDs with assigned gender (M/F):")
print(gender_attr_df["gender"].value_counts(dropna=False))



#Phase 1 – Loading and Merging CSV Files

def load_dataset(csv_name):
    df = pd.read_csv(os.path.join(DATA_PATH, csv_name))
    rate_cols = [col for col in df.columns if 'rate' in col]
    df['MOS'] = df[rate_cols].mean(axis=1)

    df["id"] = df["id"].astype(int)

    merged = df.merge(gender_attr_df, on="id", how="left")

    if merged["gender"].isna().any():
        print(f"WARNING: some IDs from {csv_name} do not have an assigned gender (gender=None)")

    return merged[['id', 'subset_id', 'MOS', 'gender']]


train_df = load_dataset("train.csv")
valid_df = load_dataset("valid.csv")
test_df = load_dataset("test.csv")

print("Phase 1 completed: CSV files have been loaded and merged.")
input("➜ Press Enter to continue to Phase 2 (MOS computation and data preparation)...")


#Phase 2 – Preparation of File Lists and MOS Scores

def prepare_file_list(df):
    file_list, mos_list, gender_list = [], [], []
    for _, row in df.iterrows():
        wav_path = os.path.join(AUDIO_PATH, f"{int(row['id']):04d}.wav")
        if os.path.exists(wav_path):
            file_list.append(wav_path)
            mos_list.append(row["MOS"])
            gender_list.append(row["gender"])
    return file_list, np.array(mos_list), np.array(gender_list)

train_files, train_mos, train_gender = prepare_file_list(train_df)
valid_files, valid_mos, valid_gender = prepare_file_list(valid_df)
test_files,  test_mos,  test_gender  = prepare_file_list(test_df)

print(f"Training samples: {len(train_files)}, Validation: {len(valid_files)}, Test: {len(test_files)}")


#Histogram of MOS Scores
all_mos = np.concatenate([train_mos, valid_mos, test_mos])
plt.figure(figsize=(7,4))
sns.histplot(all_mos, bins=10, kde=True, color='skyblue', edgecolor='black')
plt.title("Distribution of MOS Scores")
plt.xlabel("Mean Opinion Score (MOS)")
plt.ylabel("Number of Samples")
plt.tight_layout()
plt.show()


all_mos_demo = np.concatenate([train_mos, valid_mos, test_mos])
ref_stats_path = os.path.join(RESULTS_PATH, "reference_mos_stats.npz")
np.savez(ref_stats_path,
         all_mos=all_mos_demo,
         mean_mos=np.mean(all_mos_demo),
         std_mos=np.std(all_mos_demo))
print(f"Reference MOS statistics saved to: {ref_stats_path}")

print("Phase 2 completed: MOS scores have been prepared and linked to WAV files.")
input("➜ Press Enter to continue to Phase 3 (audio feature extraction)...")


#Phase 3 – Feature Extraction (MFCC, F0, Centroid, Pitch Variance)

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13), axis=1)
    f0, _, _ = librosa.pyin(y, fmin=50, fmax=400)
    f0_mean = np.nanmean(f0)
    f0_std = np.nanstd(f0)
    centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    features = np.hstack([mfcc, f0_mean, f0_std, centroid])
    return features

def create_feature_matrix(file_list):
    feats = [extract_features(f) for f in file_list]
    return np.array(feats)

print("Extracting features...")
X_train = create_feature_matrix(train_files)
X_valid = create_feature_matrix(valid_files)
X_test = create_feature_matrix(test_files)

np.save(os.path.join(RESULTS_PATH, "features_train.npy"), X_train)
np.save(os.path.join(RESULTS_PATH, "features_valid.npy"), X_valid)
np.save(os.path.join(RESULTS_PATH, "features_test.npy"), X_test)

print("Phase 3 completed: Features have been extracted and saved.")
input("➜ Press Enter to continue to Phase 4 (SVR model training)...")


#Phase 4 – SVR Model Training

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)

#Replacing NaN Values with Zeros
X_train_scaled = np.nan_to_num(X_train_scaled)
X_valid_scaled = np.nan_to_num(X_valid_scaled)
X_test_scaled = np.nan_to_num(X_test_scaled)

svr_model = SVR(kernel='rbf', C=10, epsilon=0.1)
svr_model.fit(X_train_scaled, train_mos)

pred_valid = svr_model.predict(X_valid_scaled)
pred_test = svr_model.predict(X_test_scaled)

print("Phase 4 completed: SVR model has been trained.")
input("➜ Press Enter to continue to Phase 5 (CNN+LSTM model training)...")


#Phase 5 – CNN+LSTM Model Training

def audio_to_mel(file_path, max_len=128):
    y, sr = librosa.load(file_path, sr=16000)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    
    m = np.mean(mel_db)
    s = np.std(mel_db) + 1e-8
    mel_db = (mel_db - m) / s
    #fixed input width for training
    if mel_db.shape[1] < max_len:
        pad = max_len - mel_db.shape[1]
        mel_db = np.pad(mel_db, ((0, 0), (0, pad)), mode='constant')
    else:
        mel_db = mel_db[:, :max_len]
    return mel_db

def create_mel_dataset(file_list):
    return np.array([audio_to_mel(f) for f in file_list])

X_train_mel = create_mel_dataset(train_files)
X_valid_mel = create_mel_dataset(valid_files)
X_test_mel  = create_mel_dataset(test_files)

X_train_mel = X_train_mel[..., np.newaxis]  
X_valid_mel = X_valid_mel[..., np.newaxis]
X_test_mel  = X_test_mel[..., np.newaxis]

#Model (Functional API)
from tensorflow.keras import Input, Model

inp = Input(shape=(64,128,1))  # (freq=64, time=128, ch=1)

x = layers.Conv2D(32, (3,3), padding='same', activation='relu')(inp)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D((2,2))(x)         # -> (32,64,32)
x = layers.Dropout(0.2)(x)

x = layers.Conv2D(64, (3,3), padding='same', activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D((2,2))(x)         # -> (16,32,64)
x = layers.Dropout(0.2)(x)

x = layers.Conv2D(128, (3,3), padding='same', activation='relu')(x)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D((2,2))(x)         # -> (8,16,128)
x = layers.Dropout(0.25)(x)

# Current tensor shape: (freq=8, time=16, channels=128)
x = layers.Permute((2,1,3))(x)            # -> (time=16, freq=8, ch=128)
x = layers.Reshape((16, 8*128))(x)        # -> (time=16, feat=1024)

#Bidirectional LSTM Blocks and an Additional Dense Layer
x = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(x)
x = layers.Bidirectional(layers.LSTM(64, return_sequences=False))(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.3)(x)
out = layers.Dense(1)(x)

cnn_lstm = Model(inp, out)

cnn_lstm.compile(optimizer=AdamW(learning_rate=5e-4, weight_decay=1e-4),
                 loss='mse')


#Stability Callbacks
early = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                         patience=6,
                                         restore_best_weights=True)
rlr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss',
                                           factor=0.5,
                                           patience=3,
                                           min_lr=1e-5)

history = cnn_lstm.fit(
    X_train_mel, train_mos,
    validation_data=(X_valid_mel, valid_mos),
    epochs=20,
    batch_size=8,
    callbacks=[early, rlr],
    verbose=1
)


model_path = os.path.join(RESULTS_PATH, "cnn_lstm_mos_model.keras")
cnn_lstm.save(model_path)
print(f"Model saved to: {model_path}")


print("Phase 5 completed: CNN+LSTM model has been trained.")
input("➜ Press Enter to continue to Phase 6 (model evaluation)...")


#Phase 6 – Model Evaluation (MSE, MAE, R²)

def evaluate_model(y_true, y_pred, name):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"\n{name} METRICS:")
    print(f"MSE = {mse:.4f}")
    print(f"MAE = {mae:.4f}")
    print(f"R² = {r2:.4f}")

#Errors by MOS Ranges
error_df = pd.DataFrame({
    'True_MOS': test_mos.flatten(),
    'Predicted_MOS': cnn_lstm.predict(X_test_mel).flatten()
})

error_df['Abs_Error'] = np.abs(error_df['True_MOS'] - error_df['Predicted_MOS'])

bins = [1, 2, 3, 4, 5]
labels = ['1–2', '2–3', '3–4', '4–5']
error_df['MOS_Range'] = pd.cut(error_df['True_MOS'], bins=bins, labels=labels, include_lowest=True)

error_by_range = error_df.groupby('MOS_Range')['Abs_Error'].mean()

plt.figure(figsize=(6,4))
error_by_range.plot(kind='bar', color='salmon', edgecolor='black')
plt.title('Average Prediction Error by MOS Range')
plt.xlabel('MOS Range')
plt.ylabel('Mean Absolute Error')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_PATH, "MOS_error_by_range.png"), dpi=300)
plt.close()


evaluate_model(valid_mos, pred_valid, "SVR Valid")
evaluate_model(test_mos, pred_test, "SVR Test")

cnn_preds = cnn_lstm.predict(X_test_mel).flatten()
evaluate_model(test_mos, cnn_preds, "CNN+LSTM Test")

cnn_preds_valid = cnn_lstm.predict(X_valid_mel).flatten()
evaluate_model(valid_mos, cnn_preds_valid, "CNN+LSTM Validation")


#Continuous Error Analysis

cont_df = pd.DataFrame({
    "True_MOS": test_mos.flatten(),
    "Predicted_MOS": cnn_preds.flatten()
})
cont_df["Abs_Error"] = np.abs(cont_df["True_MOS"] - cont_df["Predicted_MOS"])

cont_df = cont_df.sort_values("True_MOS").reset_index(drop=True)
cont_df["Sample_Index"] = np.arange(len(cont_df))

plt.figure(figsize=(10, 4))
ax1 = plt.gca()

ax1.plot(
    cont_df["Sample_Index"],
    cont_df["True_MOS"],
    label="Ground truth MOS",
    linewidth=1.5
)
ax1.plot(
    cont_df["Sample_Index"],
    cont_df["Predicted_MOS"],
    label="Predicted MOS (CNN-BiLSTM)",
    linewidth=1.5,
    linestyle="--"
)
ax1.set_xlabel("Sample index (sorted by MOS)")
ax1.set_ylabel("MOS value")
ax1.set_ylim(1.0, 5.2)

ax2 = ax1.twinx()
ax2.bar(
    cont_df["Sample_Index"],
    cont_df["Abs_Error"],
    alpha=0.25,
    width=1.0,
    label="Absolute error"
)
ax2.set_ylabel("Absolute error")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", fontsize=9)

plt.title("Ground truth, predictions and absolute error across MOS values")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_PATH, "MOS_error_continuous.png"), dpi=300)
plt.close()


print("Phase 6 completed: Model evaluation has been performed.")
input("➜ Press Enter to continue to Phase 7 (Grad-CAM analysis)...")


#Phase 7 – Grad-CAM Visualization

def grad_cam(model, img):
    _ = model.predict(np.expand_dims(img, axis=0))

    #Automatically identifying the last convolutional layer
    last_conv_layer = None
    for layer in reversed(model.layers):
        if isinstance(layer, layers.Conv2D):
            last_conv_layer = layer
            break

    if last_conv_layer is None:
        raise ValueError("No Conv2D layer was found in the model!")

    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output, model.outputs]
    )

    #Computing the Grad-CAM Map
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(np.expand_dims(img, axis=0))
        loss = predictions[0]

    grads = tape.gradient(loss, conv_outputs)[0]
    weights = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.reduce_sum(tf.multiply(weights, conv_outputs[0]), axis=-1)

    cam = np.maximum(cam, 0)
    cam = cam / np.max(cam)
    return cam  


#Four Grad-CAM Figures (F/M × Attractive/Non-Attractive) 

def is_useful(idx, X_mel, min_active_cols=20, eps=1e-3):
    mel = X_mel[idx].squeeze()
    active = (mel.std(axis=0) > eps).sum()
    return active >= min_active_cols


def pick_candidates(mos_array, gender_array, target_gender, top=True, k=30):
    #Returns the index of a candidate of the requested gender with either a high (top) or low (bottom) MOS score.
    #It considers at most the first k samples sorted by MOS and selects the first “usable” one (i.e., with sufficiently active spectral content).
    gender_mask = (gender_array == target_gender)
    indices = np.where(gender_mask)[0]

    if len(indices) == 0:
        return None  

    sorted_local = indices[np.argsort(mos_array[indices])]
    if top:
        sorted_local = sorted_local[::-1]  

    for idx in sorted_local[:k]:
        if is_useful(idx, X_test_mel):
            return idx

    return None


#selecting: female/male, attractive (top MOS) i non-attractive (bottom MOS)
female_high_idx = pick_candidates(test_mos, test_gender, target_gender='F', top=True)
female_low_idx  = pick_candidates(test_mos, test_gender, target_gender='F', top=False)
male_high_idx   = pick_candidates(test_mos, test_gender, target_gender='M', top=True)
male_low_idx    = pick_candidates(test_mos, test_gender, target_gender='M', top=False)

print(f"Selected Indices -> "
      f"F high: {female_high_idx}, F low: {female_low_idx}, "
      f"M high: {male_high_idx}, M low: {male_low_idx}")

if female_high_idx is None:
    female_high_idx = int(np.argmax(test_mos[test_gender == 'F']))
if female_low_idx is None:
    female_low_idx = int(np.argmin(test_mos[test_gender == 'F']))
if male_high_idx is None:
    male_high_idx = int(np.argmax(test_mos[test_gender == 'M']))
if male_low_idx is None:
    male_low_idx = int(np.argmin(test_mos[test_gender == 'M']))

mel_f_high = X_test_mel[female_high_idx]
mel_f_low  = X_test_mel[female_low_idx]
mel_m_high = X_test_mel[male_high_idx]
mel_m_low  = X_test_mel[male_low_idx]

cam_f_high = grad_cam(cnn_lstm, mel_f_high)
cam_f_low  = grad_cam(cnn_lstm, mel_f_low)
cam_m_high = grad_cam(cnn_lstm, mel_m_high)
cam_m_low  = grad_cam(cnn_lstm, mel_m_low)

def norm_cam(cam):
    cam = np.maximum(cam, 0)
    return cam / (np.max(cam) + 1e-8)

cam_f_high = norm_cam(cam_f_high)
cam_f_low  = norm_cam(cam_f_low)
cam_m_high = norm_cam(cam_m_high)
cam_m_low  = norm_cam(cam_m_low)

def trim_spectrogram(mel, min_keep_ratio=0.7):
    mel_sq = mel.squeeze()
    orig_len = mel_sq.shape[1]

    col_energy = np.abs(mel_sq).mean(axis=0)

    if np.max(col_energy) <= 0:
        return mel_sq  

    thr = 0.03 * np.max(col_energy)
    active = col_energy > thr

    if not active.any():
        return mel_sq

    last_valid = np.where(active)[0][-1]

    min_len = int(min_keep_ratio * orig_len)
    if last_valid + 1 < min_len:
        return mel_sq

    return mel_sq[:, :last_valid+1]


def plot_cam(ax, mel, cam, title):
    mel_trim = trim_spectrogram(mel)
    time_length = mel_trim.shape[1]

    librosa.display.specshow(
        mel_trim,
        sr=16000,
        x_axis='time',
        y_axis='mel',
        ax=ax
    )
    im = ax.imshow(
        cam[:, :time_length],
        cmap='jet',
        alpha=0.45,
        extent=[0, time_length, 0, mel_trim.shape[0]],
        origin='lower',
        aspect='auto'
    )
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel("Time")
    ax.set_ylabel("Hz")
    return im


fig = plt.figure(figsize=(16, 8))
gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 0.05], wspace=0.25, hspace=0.35)

fig.suptitle("Grad-CAM visualization across gender and attractiveness levels",
             fontsize=16, fontweight="bold")

ax_f_high = fig.add_subplot(gs[0, 0])
ax_f_low  = fig.add_subplot(gs[0, 1])
ax_m_high = fig.add_subplot(gs[1, 0])
ax_m_low  = fig.add_subplot(gs[1, 1])

cax = fig.add_subplot(gs[:, 2])

#Female – attractive
im1 = plot_cam(
    ax_f_high,
    mel_f_high,
    cam_f_high,
    f"Female – Attractive (MOS={test_mos[female_high_idx]:.2f})"
)

#Female – non-attractive
im2 = plot_cam(
    ax_f_low,
    mel_f_low,
    cam_f_low,
    f"Female – Non-Attractive (MOS={test_mos[female_low_idx]:.2f})"
)

#Male – attractive
im3 = plot_cam(
    ax_m_high,
    mel_m_high,
    cam_m_high,
    f"Male – Attractive (MOS={test_mos[male_high_idx]:.2f})"
)

#Male – non-attractive
im4 = plot_cam(
    ax_m_low,
    mel_m_low,
    cam_m_low,
    f"Male – Non-Attractive (MOS={test_mos[male_low_idx]:.2f})"
)

cbar = fig.colorbar(im4, cax=cax)
cbar.set_label("Grad-CAM intensity", fontsize=12)

plt.show()

print("Phase 7 completed: 4 Grad-CAM visualizations (F/M × attractive/non-attractive) have been displayed.")
