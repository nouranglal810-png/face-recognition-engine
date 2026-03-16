# 🧠 FaceScan AI - Face Recognition System

A complete face recognition ML system with a beautiful web interface and REST API.

## Features

- 🔍 **Face Recognition** - Identify faces against a registered database
- ➕ **Face Registration** - Add new people to the database with photos
- ⚖️ **Face Comparison** - Compare two photos to see if they're the same person
- 👥 **Database Management** - View, manage, and rebuild face data
- 🎨 **Beautiful UI** - Dark glassmorphism design with animations
- 🔗 **REST API** - Full API for integration with other apps

## Tech Stack

- **ML Model**: dlib's deep learning face recognition (99.38% accuracy on LFW benchmark)
- **Backend**: Python, Flask, face_recognition library
- **Frontend**: Vanilla HTML/CSS/JS with modern design
- **Image Processing**: OpenCV, Pillow

## Setup Instructions

### 1. Install Visual Studio Build Tools (Windows - Required for dlib)

dlib requires C++ compilation. Install Visual Studio Build Tools:
- Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
- During installation, select **"Desktop development with C++"**

### 2. Install CMake

```bash
pip install cmake
```

### 3. Install Python Dependencies

```bash
cd "d:\Desktop\face scane"
pip install -r requirements.txt
```

> **Note**: Installing `dlib` may take 5-10 minutes as it compiles from source.

### 4. Run the Server

```bash
python app.py
```

### 5. Open in Browser

Visit: **http://localhost:5000**

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check & stats |
| POST | `/api/detect` | Detect faces (no recognition) |
| POST | `/api/recognize` | Recognize faces against database |
| POST | `/api/register` | Register a new face |
| POST | `/api/compare` | Compare two face images |
| GET | `/api/faces` | List all registered faces |
| DELETE | `/api/faces/<name>` | Delete a person's data |
| POST | `/api/reload` | Rebuild face encodings |

## How to Use

### Register a Face
1. Go to "Register" tab
2. Enter the person's name
3. Upload a clear, front-facing photo
4. Click "Register Face"

### Recognize Faces
1. Go to "Recognize" tab
2. Upload a photo with faces
3. Click "Recognize Faces"
4. See results with confidence scores

### Compare Faces
1. Go to "Compare" tab
2. Upload two photos
3. Click "Compare Faces"
4. See if they're the same person

## Project Structure

```
face scane/
├── app.py              # Flask API server
├── face_engine.py      # Core ML engine
├── config.py           # Configuration
├── requirements.txt    # Python dependencies
├── known_faces/        # Registered face images (auto-created)
├── uploads/            # Temporary uploads (auto-created)
├── static/
│   ├── index.html      # Web interface
│   ├── style.css       # Styling
│   └── script.js       # Frontend logic
└── README.md           # This file
```

## Model Details

The system uses **dlib's ResNet-based face recognition model**:
- 128-dimensional face encoding
- 99.38% accuracy on the Labeled Faces in the Wild (LFW) benchmark
- HOG-based face detection (CPU-friendly)
- Optional CNN-based detection for higher accuracy (requires GPU)
