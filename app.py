"""
Flask REST API for Face Recognition System
Provides endpoints for face detection, recognition, registration, comparison and management.
"""

import os
import uuid
import base64
import numpy as np
import cv2
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename

import config
from face_engine import FaceEngine

# Initialize Flask app
app = Flask(__name__, static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = config.MAX_CONTENT_LENGTH
CORS(app)

# Initialize Face Engine
engine = FaceEngine()


def allowed_file(filename):
    """Check if file extension is allowed."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in config.ALLOWED_EXTENSIONS


def save_upload(file) -> str:
    """Save an uploaded file and return its path."""
    filename = secure_filename(file.filename)
    unique_name = f"{uuid.uuid4().hex}_{filename}"
    filepath = os.path.join(config.UPLOAD_DIR, unique_name)
    file.save(filepath)
    return filepath


def cleanup_file(filepath):
    """Remove a temporary uploaded file."""
    try:
        if filepath and os.path.exists(filepath):
            os.remove(filepath)
    except Exception:
        pass


# ==========================================
# API ROUTES
# ==========================================

@app.route("/")
def index():
    """Serve the web interface."""
    return send_from_directory("static", "index.html")


@app.route("/<path:filename>")
def serve_static_root(filename):
    """Serve static files from root path for relative URL compatibility.
    This allows index.html to use ./style.css and ./script.js
    which works both when opened directly (file://) and via Flask server.
    """
    import os
    static_path = os.path.join(app.static_folder, filename)
    if os.path.isfile(static_path):
        return send_from_directory("static", filename)
    # Fall through to 404 for non-existent files
    from flask import abort
    abort(404)


@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "ok",
        "service": "Face Recognition API",
        "version": "1.0.0",
        "known_faces": len(engine.known_face_names),
    })


@app.route("/api/detect", methods=["POST"])
def detect_faces():
    """
    Detect faces in an uploaded image (no recognition).
    
    POST: multipart/form-data with 'image' field
    Returns: JSON with face locations
    """
    if "image" not in request.files:
        return jsonify({"success": False, "message": "No image file provided."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"success": False, "message": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"success": False, "message": "Invalid file type."}), 400

    filepath = save_upload(file)
    try:
        result = engine.detect_faces(filepath)
        return jsonify(result)
    finally:
        cleanup_file(filepath)


@app.route("/api/recognize", methods=["POST"])
def recognize_faces():
    """
    Recognize faces in an uploaded image against known database.
    
    POST: multipart/form-data with 'image' field
    Returns: JSON with recognized faces, names, and confidence
    """
    if "image" not in request.files:
        return jsonify({"success": False, "message": "No image file provided."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"success": False, "message": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"success": False, "message": "Invalid file type."}), 400

    filepath = save_upload(file)
    try:
        result = engine.recognize_faces(filepath)
        return jsonify(result)
    finally:
        cleanup_file(filepath)


@app.route("/api/recognize_base64", methods=["POST"])
def recognize_faces_base64():
    """
    Recognize faces using a base64 encoded image instead of a file upload.
    Used for live webcam feed.
    
    POST: JSON with 'image_base64' string
    Returns: JSON with recognized faces, names, and confidence
    """
    data = request.json
    if not data or "image_base64" not in data:
        return jsonify({"success": False, "message": "No base64 image provided."}), 400
        
    base64_str = data["image_base64"]
    if "," in base64_str:
        base64_str = base64_str.split(",")[1]
        
    try:
        image_data = base64.b64decode(base64_str)
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({"success": False, "message": "Invalid image data."}), 400
            
        result = engine.recognize_faces_image(image)
        return jsonify(result)
    except Exception as e:
        return jsonify({"success": False, "message": f"Error decoding image: {str(e)}"}), 500


@app.route("/api/register", methods=["POST"])
def register_face():
    """
    Register a new face in the database.
    
    POST: multipart/form-data with 'image' and 'name' fields
    Returns: JSON with registration status
    """
    if "image" not in request.files:
        return jsonify({"success": False, "message": "No image file provided."}), 400

    if "name" not in request.form or not request.form["name"].strip():
        return jsonify({"success": False, "message": "Name is required."}), 400

    file = request.files["image"]
    name = request.form["name"].strip()

    if file.filename == "":
        return jsonify({"success": False, "message": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"success": False, "message": "Invalid file type."}), 400

    filepath = save_upload(file)
    try:
        result = engine.add_face(name, filepath)
        return jsonify(result)
    finally:
        cleanup_file(filepath)


@app.route("/api/compare", methods=["POST"])
def compare_faces():
    """
    Compare two face images to check if they are the same person.
    
    POST: multipart/form-data with 'image1' and 'image2' fields
    Returns: JSON with comparison result and confidence
    """
    if "image1" not in request.files or "image2" not in request.files:
        return jsonify({"success": False, "message": "Two images (image1, image2) are required."}), 400

    file1 = request.files["image1"]
    file2 = request.files["image2"]

    if not allowed_file(file1.filename) or not allowed_file(file2.filename):
        return jsonify({"success": False, "message": "Invalid file type."}), 400

    filepath1 = save_upload(file1)
    filepath2 = save_upload(file2)
    try:
        result = engine.compare_faces(filepath1, filepath2)
        return jsonify(result)
    finally:
        cleanup_file(filepath1)
        cleanup_file(filepath2)


@app.route("/api/faces", methods=["GET"])
def list_faces():
    """
    List all registered faces in the database.
    
    GET: No parameters needed
    Returns: JSON with list of people and their sample count
    """
    result = engine.list_known_faces()
    return jsonify(result)


@app.route("/api/faces/<name>", methods=["DELETE"])
def delete_face(name):
    """
    Delete a person from the face database.
    
    DELETE: Person name in URL
    Returns: JSON with deletion status
    """
    result = engine.delete_face(name)
    return jsonify(result)


@app.route("/api/reload", methods=["POST"])
def reload_encodings():
    """
    Force rebuild of face encodings from known_faces directory.
    Useful after manually adding/removing images.
    """
    engine._rebuild_encodings()
    return jsonify({
        "success": True,
        "message": f"Reloaded. {len(engine.known_face_names)} face(s) in database.",
        "total_faces": len(engine.known_face_names)
    })


# ==========================================
# ERROR HANDLERS
# ==========================================

@app.errorhandler(413)
def too_large(e):
    return jsonify({"success": False, "message": "File too large. Maximum size: 16MB."}), 413


@app.errorhandler(404)
def not_found(e):
    return jsonify({"success": False, "message": "Endpoint not found."}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"success": False, "message": "Internal server error."}), 500


# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Face Recognition API Server")
    print("=" * 60)
    print(f"  Engine: MediaPipe + HOG Feature Encoding")
    print(f"  Similarity Threshold: {config.SIMILARITY_THRESHOLD}")
    print(f"  Known faces: {len(engine.known_face_names)}")
    print(f"  Server: http://localhost:{config.FLASK_PORT}")
    print("=" * 60)

    app.run(
        host=config.FLASK_HOST,
        port=config.FLASK_PORT,
        debug=config.FLASK_DEBUG,
    )
