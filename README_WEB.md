# 🌐 VSL400 AI Recognition Web Application

## Overview

This document describes the web application developed for the **VSL400 Vietnamese Sign Language Word-Level Recognition** system.

The web application provides an interactive interface that allows users to upload Vietnamese Sign Language videos and receive AI-based recognition results through a user-friendly dashboard.

The system integrates the deep learning recognition pipeline into a Flask-based web platform.

---

# 🚀 Web Application Features

## 1. Video Upload & Processing

The application supports:

- Upload Vietnamese Sign Language videos (`.mp4`)
- Drag and drop video upload
- Video format validation
- Video preview before prediction
- Automatic video processing for AI inference


---

## 2. AI Prediction Interface

After uploading a video, the system performs recognition and displays:

### Predicted Gloss

The recognized Vietnamese Sign Language word.

Example:

```
Gloss:
Bóng chuyền
```


### Confidence Score

The prediction probability of the model.

Example:

```
Confidence:
32.11%
```


### Top-5 Predictions

The system provides the top five predicted glosses with confidence scores:

```
1. Bóng chuyền     32.11%
2. Bóng đá         18.54%
3. Chạy            12.30%
4. Đá bóng          8.75%
5. Đi bộ            6.20%
```


---

# 🎨 User Interface

The web interface is designed with a modern AI dashboard style.

Features include:

- Responsive web design
- Modern gradient background
- Glassmorphism card design
- Dark mode / Light mode
- Smooth animation effects
- Interactive components
- Video preview
- Prediction visualization


---

# 🏗️ System Architecture


```
+----------------+
|     User       |
+----------------+
        |
        |
        v
+----------------+
| Web Interface  |
| HTML/CSS/JS    |
+----------------+
        |
        |
        v
+----------------+
| Flask Backend  |
|    app.py      |
+----------------+
        |
        |
        v
+----------------+
| Video Upload   |
| Processing     |
+----------------+
        |
        |
        v
+----------------+
| Feature        |
| Extraction     |
| MediaPipe      |
+----------------+
        |
        |
        v
+----------------+
| AI Recognition |
| Transformer    |
| Model          |
+----------------+
        |
        |
        v
+----------------+
| Prediction     |
| Result         |
+----------------+
```


---

# 🧠 AI Pipeline Integration

The web application connects with the AI inference pipeline:

## Input

```
Vietnamese Sign Language Video
```

↓

## Feature Extraction

Extract human movement information:

- Body pose landmarks
- Left hand landmarks
- Right hand landmarks
- Temporal motion features


Using:

- MediaPipe
- OpenCV


↓

## Recognition Model

The extracted features are processed by the trained deep learning model:

- Transformer Encoder
- Attention mechanism
- Sequence modeling


↓

## Output

```
Predicted Gloss
+
Confidence Score
```


---

# 📂 Web Project Structure


```
vsl-recognition/

│
├── app.py
│   Flask server and web routing
│
├── inference.py
│   AI inference pipeline
│
├── templates/
│   │
│   ├── index.html
│   │   Upload page
│   │
│   └── result.html
│       Prediction result page
│
├── static/
│   │
│   ├── style.css
│   │   Web interface styling
│   │
│   └── script.js
│       Frontend interaction
│
├── uploads/
│   Uploaded user videos
│
└── README_WEB.md
```


---

# ⚙️ Installation


## Requirements

Recommended:

```
Python 3.11
```


Install dependencies:


```bash
pip install -r requirements.txt
```


---

# ▶️ Run Web Application


Start Flask server:

```bash
python app.py
```


Open browser:

```
http://127.0.0.1:5000
```


---

# 📖 User Workflow


1. Access the web application.

2. Upload a Vietnamese Sign Language video.

3. Preview the uploaded video.

4. Click:

```
Predict Sign
```

5. The AI model analyzes the video.

6. View:

- Predicted gloss
- Confidence score
- Top-5 predictions
- Recognition history


---

# 🛠️ Technologies Used


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

# 🔮 Future Development


Future improvements:

- Real-time webcam sign recognition
- Sentence-level sign language translation
- Cloud deployment with GPU acceleration
- User authentication system
- Mobile application
- Model optimization for faster inference


---

# 👨‍💻 Contribution

This web application extends the original VSL400 recognition project by adding:

- Complete Flask web interface
- Video upload and prediction workflow
- AI result visualization
- Modern responsive UI
- Prediction history management


---

# 📌 Note

This project is developed for educational and research purposes in Vietnamese Sign Language Recognition.
