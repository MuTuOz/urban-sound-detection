# Urban Acoustic Event Classification with CNNs

This repository contains Ozyegin University CS466 Introduction to Deep Learning final project:  
**Convolutional Neural Networks for Urban Acoustic Event Classification Using a Self-Collected Dataset**

The project focuses on classifying short urban/environmental audio clips using traditional audio features and deep learning models. We collected and organized an 8-class sound dataset, converted audio clips into log-mel spectrograms, and trained CNN-based classifiers. We also compared the CNN approach against an MFCC + MLP baseline and analyzed the effects of learning rate, dropout, and CNN depth.

---

## Project Overview

Environmental sound classification is the task of identifying the category of a sound event from an audio recording. Unlike speech recognition, this project focuses on general environmental sounds such as traffic, sirens, footsteps, construction noise, crowd chatter, public transport, birds, and wind/trees.

The main goals of this project were:

- Build an original urban acoustic event dataset.
- Train a baseline model using MFCC features and an MLP classifier.
- Train CNN models using log-mel spectrograms.
- Compare traditional handcrafted features with learned CNN features.
- Analyze the effects of learning rate, dropout, and architecture depth.

---

## Dataset

The final dataset contains **1219 audio clips**, each approximately **2 seconds** long.

Most clips were recorded using a **Zoom H4essential portable field recorder**, and a small number of publicly available web clips were used to supplement classes where field recordings were limited.

### Classes

| Class | Number of Clips |
|---|---:|
| bird_sounds | 175 |
| construction | 150 |
| crowd_chatter | 150 |
| footsteps | 150 |
| public_transport | 150 |
| siren | 150 |
| traffic | 150 |
| wind_trees | 144 |
| **Total** | **1219** |

### Dataset Split

The dataset was split into:

| Split | Number of Clips |
|---|---:|
| Train | 853 |
| Validation | 183 |
| Test | 183 |

---

## Preprocessing

All audio files were standardized before training.

Preprocessing steps:

1. Load audio using `librosa`
2. Convert to mono
3. Resample to 16 kHz
4. Pad or crop to exactly 2 seconds
5. Convert to 64-bin log-mel spectrograms
6. Normalize spectrograms before CNN training

For the MFCC baseline, 40 MFCC coefficients were extracted from each clip. The mean and standard deviation of each coefficient were concatenated, resulting in an 80-dimensional feature vector.

---

## Models

### 1. MFCC + MLP Baseline

This model uses handcrafted audio features.

Pipeline:

```text
Audio clip → MFCC extraction → mean/std pooling → MLP classifier
