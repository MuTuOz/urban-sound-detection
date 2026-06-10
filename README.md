## Project Overview

Environmental sound classification is the task of identifying the category of a sound event from an audio recording. Unlike speech recognition, this project focuses on general environmental sounds such as traffic, sirens, footsteps, construction noise, crowd chatter, public transport, birds, and wind/trees.

The original project focused on building and evaluating CNN-based classifiers on a self-collected 8-class urban acoustic event dataset. In addition to the offline deep learning experiments, the project was later extended into a **web-based audio classification system** where users can upload or record an audio clip, receive class probabilities, and provide feedback about the true sound category.

The current system supports:

* Classifying short urban/environmental audio clips through a web interface.
* Showing percentage-based prediction results for eight sound classes.
* Collecting user feedback after each prediction.
* Allowing an admin to review uploaded sounds and correct labels.
* Performing background retraining without stopping the running web service.
* Keeping the previous active model available while a new model is being trained.
* Updating the dataset cumulatively with admin-approved feedback.
* Using a CNN model together with a transfer-learning/ensemble approach for stronger predictions.

The main goals of this project are:

* Build an original urban acoustic event dataset.
* Train a baseline model using MFCC features and an MLP classifier.
* Train CNN models using log-mel spectrograms.
* Compare traditional handcrafted features with learned CNN features.
* Analyze the effects of learning rate, dropout, and architecture depth.
* Deploy the trained model inside a web-based prediction system.
* Continuously improve the model using feedback and retraining.
* Extend the system with transfer learning and ensemble-based prediction.

---

## Dataset

The original experimental dataset contains **1219 audio clips**, each approximately **2 seconds** long.

Most clips were recorded using a **Zoom H4essential portable field recorder**, and a small number of publicly available web clips were used to supplement classes where field recordings were limited.

In the web-based version, the dataset can be expanded over time. New audio samples can be added through:

* Manually collected recordings
* Publicly available audio sources
* Freesound-based dataset expansion
* User-uploaded feedback samples
* Admin-approved corrected labels

Admin-approved feedback samples can be added into the base dataset so that future retraining becomes cumulative.

### Classes

| Class            | Number of Clips in Original Dataset |
| ---------------- | ----------------------------------: |
| bird_sounds      |                                 175 |
| construction     |                                 150 |
| crowd_chatter    |                                 150 |
| footsteps        |                                 150 |
| public_transport |                                 150 |
| siren            |                                 150 |
| traffic          |                                 150 |
| wind_trees       |                                 144 |
| **Total**        |                            **1219** |

### Dataset Split

The original offline experiment used the following split:

| Split      | Number of Clips |
| ---------- | --------------: |
| Train      |             853 |
| Validation |             183 |
| Test       |             183 |

In the deployed system, the base dataset can grow over time as more approved samples are added. Therefore, the number of training samples may change after retraining.

---

## Preprocessing

All audio files are standardized before training and prediction.

Preprocessing steps:

1. Load audio using `librosa`
2. Convert to mono
3. Resample to 16 kHz
4. Pad or crop to exactly 2 seconds
5. Convert to 64-bin log-mel spectrograms
6. Normalize spectrograms before CNN training

For uploaded user audio, the system can also scan longer recordings using overlapping 2-second windows. This prevents the model from relying only on the first two seconds of an uploaded file, which is useful when the important sound event starts later in the recording.

For the MFCC baseline, 40 MFCC coefficients were extracted from each clip. The mean and standard deviation of each coefficient were concatenated, resulting in an 80-dimensional feature vector.

---

## Models

### 1. MFCC + MLP Baseline

This model uses handcrafted audio features.

Pipeline:

```text
Audio clip → MFCC extraction → mean/std pooling → MLP classifier
```

The MFCC baseline was used as a traditional reference model before training CNN-based models. It achieved strong performance and showed that the dataset is learnable even with compact handcrafted features.

### 2. Log-Mel CNN

The main deep learning approach uses log-mel spectrograms as 2D inputs to a convolutional neural network.

Pipeline:

```text
Audio clip → log-mel spectrogram → CNN → softmax class probabilities
```

The CNN learns local time-frequency patterns from spectrograms. This is useful for distinguishing sounds such as sirens, footsteps, traffic, wind, and crowd chatter.

### 3. Deeper Log-Mel CNN

The best-performing offline model was a deeper CNN with four convolutional blocks:

```text
Conv32 → Conv64 → Conv128 → Conv256 → Global Average Pooling → Dense128 → Softmax
```

This model achieved the best macro-F1 score in the original experiments. It was later used as the main model for the web-based prediction system.

### 4. Transfer Learning / Ensemble Extension

To improve robustness on real user-uploaded audio, the system was extended with a transfer-learning based prediction layer.

The current advanced prediction approach can combine:

```text
Log-mel CNN prediction
+
Pretrained audio embedding classifier
+
Weighted ensemble
```

This ensemble-style approach helps the system become less dependent on a single CNN output. It is especially useful for public audio files, MP3 files, phone recordings, and sounds that differ from the original recording setup.

---

## Web-Based System

The project was extended into a web-based audio classification system.

Main workflow:

```text
User uploads or records an audio clip
↓
System preprocesses the audio
↓
Active model predicts class probabilities
↓
Prediction result is shown to the user
↓
User provides the true label as feedback
↓
Admin reviews the uploaded sound and label
↓
Approved samples are used in background retraining
```

The web system includes:

* Audio upload support
* Audio recording support
* Percentage-based prediction results
* User feedback collection
* Admin panel
* Uploaded audio review
* Admin label correction
* Background retraining
* Model version tracking
* Active model protection during training

When retraining starts, the old active model remains available for prediction. The new model only becomes active after training finishes successfully.

---

## Feedback and Cumulative Retraining

The system supports feedback-based model improvement.

After a user uploads a sound and provides the true label, the sample is stored for admin review. The admin can listen to the uploaded sound, correct the label if necessary, and select samples for retraining.

Retraining behavior:

```text
Base dataset
+
Admin-approved feedback samples
↓
Background retraining
↓
Candidate model evaluation
↓
New model activation if validation performance is acceptable
```

The retraining process is cumulative. Approved feedback samples can be copied into the base dataset, so future retraining uses an increasingly larger and more representative dataset.

This design prevents the web service from becoming unavailable during model training.

---

## Results

In the original offline experiments, the best-performing model was the deeper log-mel CNN.

| Model              | Test Accuracy | Test Macro-F1 |
| ------------------ | ------------: | ------------: |
| MFCC + MLP         |        0.9180 |        0.9108 |
| Deeper Log-Mel CNN |        0.9290 |        0.9293 |

The results show that CNNs trained on log-mel spectrograms can slightly outperform traditional MFCC-based features on this dataset.

The most important experimental finding was the effect of learning rate. A learning rate of `1e-3` caused unstable training, while reducing the learning rate to `1e-4` produced stable and accurate CNN models.

---

## Current System Status

The current version is no longer only an offline notebook-style experiment. It is a deployable web-based system with:

* A trained CNN model
* User audio upload
* Audio recording
* Prediction percentages
* Feedback collection
* Admin review
* Background retraining
* Cumulative dataset growth
* Model versioning
* Transfer-learning/ensemble extension

This makes the project closer to a real-world machine learning application where the model can improve over time through supervised feedback.


go app. directory and  docker compose up --build -d and then go localhost 8000 use envexample for admin token
