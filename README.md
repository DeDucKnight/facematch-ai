# FaceMatch-AI

A minimal face matching API using FastAPI and DeepFace, designed as a prototype for an immigration system. It supports face verification, embedding comparison, and enrollment of new users.

---

## 🚀 Features

- Upload two face images and check if they match
- Uses `Facenet` for face embeddings
- Optional detectors: RetinaFace (default), MTCNN, OpenCV
- FastAPI-powered API with Swagger UI
- Dummy enrollment system (in-memory face embedding storage)

---

## 📦 Installation

### 1. Clone the repo
```bash
git clone https://github.com/your-username/facematch-ai.git
cd facematch-ai
```

### 2. Create virtual environment
```bash
python -m venv venv
venv\Scripts\activate   # Windows
# OR
source venv/bin/activate  # macOS/Linux
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

▶️ Run the Server
```bash
uvicorn main:app --reload
```

📸 API Endpoints
POST /match
Accepts 2 face images

Returns match status + confidence score

POST /enroll
Accepts user_id + image

Stores face embedding in memory for later use


📁 Notes & Assumptions
Uses DeepFace’s Facenet model by default

Uses retinaface detector (you can change to mtcnn or opencv in utils.py)

This is a prototype — embeddings are stored in-memory (user_db dict)


🛠 Tech Stack
Python 3.12

FastAPI

DeepFace

Uvicorn

Pillow