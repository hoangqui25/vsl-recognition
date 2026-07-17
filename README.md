# 🇻🇳 VSL400 AI Recognition
## Vietnamese Sign Language Word-Level Recognition Web Application

A deep learning based web application for recognizing Vietnamese Sign Language (VSL) at word level from video input.

The system receives Vietnamese sign language videos from users, extracts human skeleton features, processes sequential information using deep learning models, and predicts the corresponding sign gloss.

This project provides a complete AI pipeline integrated into a modern web interface.

---

# 🌟 Project Overview

Vietnamese Sign Language Recognition is an important application that helps bridge communication between deaf and hearing communities.

This project focuses on:

- Video-based Vietnamese Sign Language Recognition
- Word-level gloss classification
- Skeleton-based feature extraction
- Transformer-based sequence modeling
- AI inference through a Flask Web Application

The final system allows users to upload a sign language video and receive an automatic prediction result.

---

# 🚀 Main Features

## 🤖 AI Recognition System

The application supports:

✅ Vietnamese Sign Language word recognition  
✅ Video input processing  
✅ Skeleton feature extraction  
✅ Transformer-based recognition model  
✅ Confidence score estimation  
✅ Top-K prediction ranking  


---

# 🌐 Web Application Features

The system provides a modern AI dashboard:

## Video Upload

Users can:

- Upload `.mp4` sign language videos
- Drag & drop videos
- Preview uploaded videos before prediction


## Prediction Result

After inference, the system displays:

- Predicted gloss
- Confidence percentage
- Top-5 prediction results
- Uploaded video preview
- Prediction history


Example:

```
Predicted Gloss:

Bóng chuyền


Confidence:

32.11%
```


---

# 📊 Dataset Information

The project is based on the VSL400 dataset.

Dataset statistics:

| Attribute | Value |
|-----------|-------|
| Videos | 74,259 |
| Gloss Classes | 400 |
| Signers | 28 |
| Views | 3 |


The dataset contains Vietnamese Sign Language videos collected from multiple signers and viewpoints.


---

# 🏗️ System Architecture


```
                 User

                  |
                  |
                  v

            Web Interface

                  |
                  |
                  v

          Flask Application

                  |
                  |
                  v

          Video Processing

                  |
                  |
                  v

        Feature Extraction

        (MediaPipe Skeleton)

                  |
                  |
                  v

       Transformer Recognition Model

                  |
                  |
                  v

          Prediction Output

                  |
                  |
                  v

             Web Result
```


---

# 🧠 AI Pipeline


## 1. Video Input

User uploads a Vietnamese Sign Language video.

Example:

```
uploads/video.mp4
```


## 2. Feature Extraction

The system extracts:

- Body pose landmarks
- Left hand landmarks
- Right hand landmarks
- Temporal motion information


Using:

- MediaPipe
- OpenCV


## 3. Sequence Recognition

Extracted features are processed by deep learning models:

- Transformer Encoder
- Attention mechanism
- Temporal feature learning


## 4. Prediction

The model outputs:

```
Gloss label
+
Confidence score
```


---

# 📂 Project Structure


```
vsl-recognition/

│
├── app.py                  # Flask web server
│
├── inference.py            # AI inference pipeline
│
├── requirements.txt        # Python dependencies
│
│
├── models/
│   ├── mediapipe/
│   │   └── holistic_landmarker.task
│   │
│   └── trained models
│
│
├── uploads/
│   └── uploaded videos
│
│
├── templates/
│   ├── index.html
│   └── result.html
│
│
├── static/
│   ├── style.css
│   └── script.js
│
│
├── history.json
│
└── README.md

```


---

# 🛠️ Installation


## Requirements

Recommended environment:

```
Python 3.11
```


Python 3.14 is not recommended because some AI libraries may not support it completely.


---

## Create Virtual Environment


```bash
python -m venv venv
```


Activate environment:


### Windows

```bash
venv\Scripts\activate
```


### Linux / MacOS

```bash
source venv/bin/activate
```


---

## Install Dependencies


```bash
pip install -r requirements.txt
```


Main libraries:

```
Flask
PyTorch
TorchVision
OpenCV
MediaPipe
NumPy
TQDM
```


---

# ▶️ Run Application


Start Flask server:


```bash
python app.py
```


The application will run at:


```
http://127.0.0.1:5000
```


Open this URL in your browser.


---

# 📖 User Guide


## Step 1

Open the website.


## Step 2

Upload Vietnamese Sign Language video.


## Step 3

Click:

```
Predict Sign
```


## Step 4

Wait for AI inference.


## Step 5

View:

- Recognized gloss
- Confidence score
- Top-5 predictions
- Skeleton information
- History records


---

# 🎨 Web Interface


The application includes:

- Modern gradient UI
- Glassmorphism design
- Responsive layout
- Dark mode
- Animation effects
- Interactive dashboard


---

# 💻 Technologies


## Backend

- Python
- Flask


## Artificial Intelligence

- PyTorch
- Transformer
- MediaPipe
- OpenCV


## Frontend

- HTML5
- CSS3
- JavaScript
- Bootstrap 5


---

# ☁️ Deployment


The application can be deployed on:


## Local Deployment

Run:

```bash
python app.py
```


Access:

```
localhost:5000
```


## Cloud Deployment

Possible platforms:

- Render
- Railway
- Hugging Face Spaces
- AWS
- Google Cloud Platform


---

# 🔮 Future Improvements


Future development directions:


## AI Improvements

- Larger Transformer models
- Better temporal modeling
- Attention visualization
- Real-time recognition


## Application Improvements

- Webcam live recognition
- Mobile application
- Automatic Vietnamese sentence translation
- Cloud GPU inference
- User accounts and authentication


---

# 👨‍💻 Author


Vietnamese Sign Language Recognition AI Project


A complete AI-powered web application for Vietnamese Sign Language Word-Level Recognition.


---

# 📜 License


This project is developed for educational and research purposes.
