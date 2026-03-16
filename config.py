"""
Configuration settings for Face Recognition System
Uses MediaPipe for face detection and a histogram-based encoding for recognition.
"""
import os

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Directory to store known face images
KNOWN_FACES_DIR = os.path.join(BASE_DIR, "known_faces")

# Directory for uploaded images (temporary)
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")

# Model encoding file (cached encodings)
ENCODINGS_FILE = os.path.join(BASE_DIR, "face_encodings.pkl")

# Face recognition similarity threshold (higher = more strict, 0-1 scale)
# Adjusted to 0.82 for the new normalized strict multi-feature algorithm
SIMILARITY_THRESHOLD = 0.82

# Minimum face detection confidence for MediaPipe
MIN_DETECTION_CONFIDENCE = 0.5

# Flask settings
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
FLASK_DEBUG = True

# Allowed image extensions
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp", "webp"}

# Max file upload size (16 MB)
MAX_CONTENT_LENGTH = 16 * 1024 * 1024

# Face encoding size for the histogram-based approach
ENCODING_SIZE = 128

# Create directories if they don't exist
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
