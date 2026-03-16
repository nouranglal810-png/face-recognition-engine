"""
Face Recognition Engine — v2.0 (Multi-Feature Robust Encoding)
Uses MediaPipe Tasks API for face detection + multi-feature encoding for recognition.

Encoding approach (improved):
- Detect face with MediaPipe Face Detector (Tasks API)
- Extract face region from the image
- Generate feature vector using:
  1. Multi-scale LBP (Local Binary Patterns) — texture invariant features
  2. Dense HOG with CLAHE preprocessing
  3. Landmark-based geometric ratios (via FaceLandmarker)
  4. Center-region histogram
- Compare using cosine similarity with multi-augmentation matching
"""

import os
import pickle
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions
from PIL import Image
import config


class FaceEngine:
    """
    Core face recognition engine using MediaPipe Tasks API + multi-feature encoding.
    """

    def __init__(self):
        # Initialize MediaPipe Face Detector (Tasks API)
        model_path = os.path.join(config.BASE_DIR, "blaze_face_short_range.tflite")

        if not os.path.exists(model_path):
            print("[FaceEngine] Downloading face detection model...")
            import urllib.request
            model_url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
            urllib.request.urlretrieve(model_url, model_path)
            print("[FaceEngine] Model downloaded.")

        options = vision.FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
        )
        self.face_detector = vision.FaceDetector.create_from_options(options)

        # Initialize MediaPipe Face Landmarker for emotions + geometric features
        landmarker_path = os.path.join(config.BASE_DIR, "face_landmarker.task")
        if not os.path.exists(landmarker_path):
            print("[FaceEngine] Downloading face landmarker model for emotions...")
            import urllib.request
            landmarker_url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            urllib.request.urlretrieve(landmarker_url, landmarker_path)
            print("[FaceEngine] Landmarker Model downloaded.")

        landmarker_options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=landmarker_path),
            output_face_blendshapes=True,
            num_faces=10,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5
        )
        self.face_landmarker = vision.FaceLandmarker.create_from_options(landmarker_options)

        # Known face data
        self.known_face_encodings = []
        self.known_face_names = []
        self._load_encodings()

    # =========================================================
    # EMOTION DETECTION
    # =========================================================

    def _get_emotion_from_blendshapes(self, blendshapes) -> str:
        """Infer basic emotion heuristically from ARKit face blendshapes."""
        scores = {b.category_name: b.score for b in blendshapes}
        
        happy = (scores.get("mouthSmileLeft", 0) + scores.get("mouthSmileRight", 0)) / 2.0
        surprise = (scores.get("jawOpen", 0) + (scores.get("eyeWideLeft", 0) + scores.get("eyeWideRight", 0)) / 2.0) / 2.0
        angry = (scores.get("browDownLeft", 0) + scores.get("browDownRight", 0)) / 2.0
        sad = (scores.get("mouthFrownLeft", 0) + scores.get("mouthFrownRight", 0) + scores.get("browInnerUp", 0)) / 3.0
        
        emotions = {
            "Happy 😊": happy * 1.5,
            "Surprised 😲": surprise * 1.2,
            "Angry 😠": angry,
            "Sad 😔": sad,
            "Neutral 😐": 0.15
        }
        
        best_emotion = max(emotions, key=lambda k: emotions[k])
        if emotions[best_emotion] < 0.15:
            return "Neutral 😐"
        return best_emotion

    # =========================================================
    # FEATURE ENCODING (v2 — Multi-Feature Robust)
    # =========================================================

    def _compute_lbp(self, gray_img, radius=1, n_points=8):
        """Compute Local Binary Pattern histogram for texture features."""
        h, w = gray_img.shape
        lbp = np.zeros((h, w), dtype=np.uint8)
        
        for i in range(radius, h - radius):
            for j in range(radius, w - radius):
                center = gray_img[i, j]
                binary = 0
                for k in range(n_points):
                    angle = 2 * np.pi * k / n_points
                    x = i + int(round(radius * np.cos(angle)))
                    y = j + int(round(radius * np.sin(angle)))
                    if gray_img[x, y] >= center:
                        binary |= (1 << k)
                lbp[i, j] = binary
        
        # Compute histogram
        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
        hist = hist.astype(np.float32)
        norm = np.linalg.norm(hist)
        if norm > 0:
            hist = hist / norm
        return hist

    def _compute_lbp_fast(self, gray_img, radius=1):
        """Fast vectorized LBP implementation using numpy."""
        h, w = gray_img.shape
        padded = cv2.copyMakeBorder(gray_img, radius, radius, radius, radius, cv2.BORDER_REFLECT)
        center = padded[radius:radius+h, radius:radius+w].astype(np.int16)
        
        # 8 neighbors
        offsets = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]
        lbp = np.zeros((h, w), dtype=np.uint8)
        
        for bit, (dy, dx) in enumerate(offsets):
            neighbor = padded[radius+dy:radius+h+dy, radius+dx:radius+w+dx].astype(np.int16)
            lbp |= ((neighbor >= center).astype(np.uint8) << bit)
        
        hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
        hist = hist.astype(np.float32)
        norm = np.linalg.norm(hist)
        if norm > 0:
            hist = hist / norm
        return hist

    def _extract_geometric_features(self, image_bgr):
        """Extract landmark-based geometric ratios using FaceLandmarker.
        These ratios are scale/position invariant — ideal for face identity."""
        try:
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            result = self.face_landmarker.detect(mp_image)
            
            if not result.face_landmarks or len(result.face_landmarks) == 0:
                return np.zeros(16, dtype=np.float32)
            
            landmarks = result.face_landmarks[0]
            h, w, _ = image_bgr.shape
            
            # Key landmark indices (MediaPipe canonical face mesh)
            # 33: left eye inner, 263: right eye inner
            # 1: nose tip, 61: left mouth corner, 291: right mouth corner
            # 199: chin bottom, 10: forehead top
            # 130: left eye outer, 359: right eye outer
            # 17: lower lip bottom, 0: upper lip top
            # 4: nose bridge, 152: chin
            
            def pt(idx):
                return np.array([landmarks[idx].x * w, landmarks[idx].y * h])
            
            def dist(p1, p2):
                return np.linalg.norm(p1 - p2)
            
            # Compute geometric ratios
            left_eye_inner = pt(33)
            right_eye_inner = pt(263)
            left_eye_outer = pt(130)
            right_eye_outer = pt(359)
            nose_tip = pt(1)
            nose_bridge = pt(4)
            left_mouth = pt(61)
            right_mouth = pt(291)
            chin = pt(152)
            forehead = pt(10)
            lower_lip = pt(17)
            upper_lip = pt(0)
            
            # Reference distance (inter-eye distance) for normalization
            eye_dist = dist(left_eye_inner, right_eye_inner)
            if eye_dist < 1:
                return np.zeros(16, dtype=np.float32)
            
            features = np.array([
                dist(left_eye_outer, right_eye_outer) / eye_dist,     # 0: eye width ratio
                dist(nose_tip, chin) / eye_dist,                       # 1: nose-to-chin
                dist(forehead, chin) / eye_dist,                       # 2: face height
                dist(left_mouth, right_mouth) / eye_dist,              # 3: mouth width
                dist(nose_tip, upper_lip) / eye_dist,                  # 4: nose-to-lip
                dist(nose_bridge, nose_tip) / eye_dist,                # 5: nose length
                dist(left_eye_inner, nose_tip) / eye_dist,             # 6: left eye to nose
                dist(right_eye_inner, nose_tip) / eye_dist,            # 7: right eye to nose
                dist(left_mouth, chin) / eye_dist,                     # 8: left mouth to chin
                dist(right_mouth, chin) / eye_dist,                    # 9: right mouth to chin
                dist(forehead, nose_tip) / eye_dist,                   # 10: forehead to nose
                dist(left_eye_outer, left_mouth) / eye_dist,           # 11: left eye to mouth
                dist(right_eye_outer, right_mouth) / eye_dist,         # 12: right eye to mouth
                dist(upper_lip, lower_lip) / eye_dist,                 # 13: lip height
                dist(left_eye_inner, left_eye_outer) / eye_dist,       # 14: left eye width
                dist(right_eye_inner, right_eye_outer) / eye_dist,     # 15: right eye width
            ], dtype=np.float32)
            
            return features
            
        except Exception as e:
            print(f"[FaceEngine] Geometric feature error: {e}")
            return np.zeros(16, dtype=np.float32)

    def _compute_face_encoding(self, face_img, full_image=None):
        """
        Compute a robust multi-feature vector for a face image.
        
        Features:
        1. Dense HOG (with CLAHE preprocessing)
        2. Multi-scale LBP texture features (scale 1 + scale 2)
        3. Center region histogram
        4. Landmark geometric ratios (if full_image provided)
        """
        # Resize to standard size 
        face_resized = cv2.resize(face_img, (96, 96))
        gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
        
        # CLAHE for better contrast normalization (much better than equalizeHist)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # Reduce noise
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # --- Feature 1: Dense HOG features ---
        win_size = (96, 96)
        block_size = (16, 16)
        block_stride = (8, 8)
        cell_size = (8, 8)
        num_bins = 9
        hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, num_bins)
        hog_features = hog.compute(gray).flatten()
        
        # --- Feature 2: Multi-scale LBP ---
        # Scale 1: fine texture (radius=1)
        gray_small = cv2.resize(gray, (48, 48))
        lbp_fine = self._compute_lbp_fast(gray_small, radius=1)
        
        # Scale 2: coarser texture (radius=2 on smaller image)
        gray_tiny = cv2.resize(gray, (32, 32))
        lbp_coarse = self._compute_lbp_fast(gray_tiny, radius=1)
        
        # --- Feature 3: Center region histogram ---
        center = gray[24:72, 24:72]
        center_hist = cv2.calcHist([center], [0], None, [32], [0, 256]).flatten()
        center_hist = center_hist.astype(np.float32)
        c_norm = np.linalg.norm(center_hist)
        if c_norm > 0:
            center_hist = center_hist / c_norm
        
        # --- Normalize individual features to prevent HOG dominance ---
        hnorm = np.linalg.norm(hog_features)
        if hnorm > 0: hog_features = hog_features / hnorm
        
        lnorm1 = np.linalg.norm(lbp_fine)
        if lnorm1 > 0: lbp_fine = lbp_fine / lnorm1
        
        lnorm2 = np.linalg.norm(lbp_coarse)
        if lnorm2 > 0: lbp_coarse = lbp_coarse / lnorm2
        
        cnorm = np.linalg.norm(center_hist)
        if cnorm > 0: center_hist = center_hist / cnorm
        
        # --- Feature 4: Geometric ratios ---
        if full_image is not None:
            geo_features = self._extract_geometric_features(full_image)
        else:
            geo_features = self._extract_geometric_features(face_img)
            
        gnorm = np.linalg.norm(geo_features)
        if gnorm > 0: geo_features = geo_features / gnorm
        
        # --- Combine all features with balanced weights ---
        encoding = np.concatenate([
            hog_features * 0.40,   # Shape
            lbp_fine * 0.20,       # Fine Texture
            lbp_coarse * 0.10,     # Coarse Texture
            center_hist * 0.05,    # Tone
            geo_features * 0.55    # Structural Geometry
        ])
        
        # Normalize to unit length for cosine similarity
        norm = np.linalg.norm(encoding)
        if norm > 0:
            encoding = encoding / norm

        return encoding.astype(np.float32)

    def _compute_augmented_encodings(self, face_img, full_image=None):
        """Generate multiple encodings with augmentations for robust matching."""
        encodings = []
        
        # Original
        encodings.append(self._compute_face_encoding(face_img, full_image))
        
        # Brightness +15%
        bright = cv2.convertScaleAbs(face_img, alpha=1.15, beta=10)
        encodings.append(self._compute_face_encoding(bright, full_image))
        
        # Brightness -15%
        dark = cv2.convertScaleAbs(face_img, alpha=0.85, beta=-10)
        encodings.append(self._compute_face_encoding(dark, full_image))
        
        return encodings

    # =========================================================
    # FACE DETECTION
    # =========================================================

    def _detect_faces_mediapipe(self, image_bgr):
        """
        Detect faces using MediaPipe Tasks API.
        Returns: List of (top, right, bottom, left) tuples in pixel coordinates
        """
        h, w, _ = image_bgr.shape
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        detection_result = self.face_detector.detect(mp_image)

        face_locations = []
        if detection_result.detections:
            for detection in detection_result.detections:
                bbox = detection.bounding_box
                x = bbox.origin_x
                y = bbox.origin_y
                bw = bbox.width
                bh = bbox.height

                margin_x = int(bw * 0.15)
                margin_y = int(bh * 0.15)

                top = max(0, y - margin_y)
                right = min(w, x + bw + margin_x)
                bottom = min(h, y + bh + margin_y)
                left = max(0, x - margin_x)

                face_locations.append((top, right, bottom, left))

        return face_locations

    # =========================================================
    # ENCODING PERSISTENCE
    # =========================================================

    def _load_encodings(self):
        """Load cached encodings from disk, or rebuild from known_faces directory."""
        if os.path.exists(config.ENCODINGS_FILE):
            try:
                with open(config.ENCODINGS_FILE, "rb") as f:
                    data = pickle.load(f)
                    self.known_face_encodings = data.get("encodings", [])
                    self.known_face_names = data.get("names", [])
                print(f"[FaceEngine] Loaded {len(self.known_face_names)} known face(s) from cache.")
                return
            except Exception as e:
                print(f"[FaceEngine] Error loading cache: {e}. Rebuilding...")

        self._rebuild_encodings()

    def _rebuild_encodings(self):
        """Scan known_faces directory and build encodings for all images."""
        self.known_face_encodings = []
        self.known_face_names = []

        if not os.path.exists(config.KNOWN_FACES_DIR):
            os.makedirs(config.KNOWN_FACES_DIR, exist_ok=True)
            print("[FaceEngine] known_faces directory created. No faces to load.")
            return

        for person_name in os.listdir(config.KNOWN_FACES_DIR):
            person_dir = os.path.join(config.KNOWN_FACES_DIR, person_name)
            if not os.path.isdir(person_dir):
                continue

            for img_file in os.listdir(person_dir):
                img_path = os.path.join(person_dir, img_file)
                ext = img_file.rsplit(".", 1)[-1].lower() if "." in img_file else ""
                if ext not in config.ALLOWED_EXTENSIONS:
                    continue

                try:
                    image = cv2.imread(img_path)
                    if image is None:
                        print(f"[FaceEngine] Could not read: {img_path}")
                        continue

                    face_locations = self._detect_faces_mediapipe(image)

                    if face_locations:
                        top, right, bottom, left = face_locations[0]
                        face_crop = image[top:bottom, left:right]

                        if face_crop.size > 0:
                            encoding = self._compute_face_encoding(face_crop, image)
                            self.known_face_encodings.append(encoding)
                            self.known_face_names.append(person_name)
                            print(f"[FaceEngine] Encoded: {person_name} <- {img_file}")
                        else:
                            print(f"[FaceEngine] Empty face crop in: {img_path}")
                    else:
                        print(f"[FaceEngine] No face found in: {img_path}")
                except Exception as e:
                    print(f"[FaceEngine] Error processing {img_path}: {e}")

        self._save_encodings()
        print(f"[FaceEngine] Total known faces: {len(self.known_face_names)}")

    def _save_encodings(self):
        """Save current encodings to disk."""
        data = {
            "encodings": self.known_face_encodings,
            "names": self.known_face_names,
        }
        with open(config.ENCODINGS_FILE, "wb") as f:
            pickle.dump(data, f)
        print("[FaceEngine] Encodings saved to cache.")

    # =========================================================
    # SIMILARITY
    # =========================================================

    def _cosine_similarity(self, enc1, enc2):
        """Compute cosine similarity between two encodings."""
        # Handle different encoding sizes (old vs new)
        min_len = min(len(enc1), len(enc2))
        e1 = enc1[:min_len]
        e2 = enc2[:min_len]
        
        dot_product = np.dot(e1, e2)
        norm1 = np.linalg.norm(e1)
        norm2 = np.linalg.norm(e2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(dot_product / (norm1 * norm2))

    # =========================================================
    # PUBLIC API
    # =========================================================

    def add_face(self, name: str, image_path: str) -> dict:
        """Add a new face to the known faces database.
        If name already exists, replaces the old encoding (one entry per person)."""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return {"success": False, "message": "Could not read the image file."}

            face_locations = self._detect_faces_mediapipe(image)

            if not face_locations:
                return {
                    "success": False,
                    "message": "No face detected in the image. Please upload a clear photo with a visible face."
                }

            if len(face_locations) > 1:
                return {
                    "success": False,
                    "message": f"Multiple faces ({len(face_locations)}) detected. Please upload a photo with only one face."
                }

            top, right, bottom, left = face_locations[0]
            face_crop = image[top:bottom, left:right]

            if face_crop.size == 0:
                return {"success": False, "message": "Face region is too small. Try a closer photo."}

            encoding = self._compute_face_encoding(face_crop, image)

            # Save image to known_faces directory
            person_dir = os.path.join(config.KNOWN_FACES_DIR, name)
            os.makedirs(person_dir, exist_ok=True)

            existing = len([f for f in os.listdir(person_dir) if os.path.isfile(os.path.join(person_dir, f))])
            ext = image_path.rsplit(".", 1)[-1].lower() if "." in image_path else "jpg"
            save_path = os.path.join(person_dir, f"{name}_{existing + 1}.{ext}")

            img = Image.open(image_path)
            img.save(save_path)

            # Remove old encodings for this name (one entry per person)
            old_indices = [i for i, n in enumerate(self.known_face_names) if n == name]
            for idx in sorted(old_indices, reverse=True):
                self.known_face_encodings.pop(idx)
                self.known_face_names.pop(idx)

            # Add the new encoding
            self.known_face_encodings.append(encoding)
            self.known_face_names.append(name)
            self._save_encodings()

            action = "updated" if old_indices else "registered"
            return {
                "success": True,
                "message": f"Face {action} successfully for '{name}'. Total images: {existing + 1}"
            }

        except Exception as e:
            return {"success": False, "message": f"Error: {str(e)}"}

    def recognize_faces(self, image_path: str) -> dict:
        """Detect and recognize faces in an image file."""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return {"success": False, "faces_found": 0, "faces": [], "message": "Could not read image."}
            return self.recognize_faces_image(image)
        except Exception as e:
            return {"success": False, "faces_found": 0, "faces": [], "message": f"Error: {str(e)}"}

    def recognize_faces_image(self, image) -> dict:
        """Detect and recognize faces AND emotions in an OpenCV image array."""
        try:
            face_locations = self._detect_faces_mediapipe(image)

            if not face_locations:
                return {
                    "success": True,
                    "faces_found": 0,
                    "faces": [],
                    "message": "No faces detected in the image."
                }
                
            # Extract Blendshapes (Emotions) via FaceLandmarker
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            landmarker_result = self.face_landmarker.detect(mp_image)
            
            h, w, _ = image.shape
            landmark_centers = []
            if landmarker_result.face_landmarks:
                for idx, landmarks in enumerate(landmarker_result.face_landmarks):
                    lx = [lm.x * w for lm in landmarks]
                    ly = [lm.y * h for lm in landmarks]
                    center_x = float(sum(lx) / len(lx))
                    center_y = float(sum(ly) / len(ly))
                    blendshapes = landmarker_result.face_blendshapes[idx] if landmarker_result.face_blendshapes else []
                    landmark_centers.append({"cx": center_x, "cy": center_y, "blendshapes": blendshapes})

            results = []
            for (top, right, bottom, left) in face_locations:
                face_crop = image[top:bottom, left:right]

                if face_crop.size == 0:
                    continue

                encoding = self._compute_face_encoding(face_crop, image)

                name = "Unknown"
                confidence = 0.0
                emotion = "Neutral 😐"

                # Match with nearest landmarker result
                box_cx = (left + right) / 2
                box_cy = (top + bottom) / 2
                best_dist = float('inf')
                best_blendshapes = None
                
                for lc in landmark_centers:
                    dist = (lc["cx"] - box_cx)**2 + (lc["cy"] - box_cy)**2
                    if dist < best_dist and dist < ((right-left)**2 + (bottom-top)**2):
                        best_dist = dist
                        best_blendshapes = lc["blendshapes"]
                
                if best_blendshapes:
                    emotion = self._get_emotion_from_blendshapes(best_blendshapes)

                if self.known_face_encodings:
                    similarities = [
                        self._cosine_similarity(encoding, known_enc)
                        for known_enc in self.known_face_encodings
                    ]

                    best_idx = np.argmax(similarities)
                    best_similarity = similarities[best_idx]

                    if best_similarity >= config.SIMILARITY_THRESHOLD:
                        name = self.known_face_names[best_idx]
                        confidence = float(f"{best_similarity * 100:.2f}")

                results.append({
                    "name": name,
                    "expression": emotion,
                    "confidence": confidence,
                    "location": {
                        "top": int(top),
                        "right": int(right),
                        "bottom": int(bottom),
                        "left": int(left),
                    }
                })

            return {
                "success": True,
                "faces_found": len(results),
                "faces": results,
                "message": f"Detected {len(results)} face(s)."
            }

        except Exception as e:
            return {"success": False, "faces_found": 0, "faces": [], "message": f"Error: {str(e)}"}

    def detect_faces(self, image_path: str) -> dict:
        """Only detect faces (no recognition)."""
        try:
            image = cv2.imread(image_path)
            if image is None:
                return {"success": False, "faces_found": 0, "faces": [], "message": "Could not read image."}

            face_locations = self._detect_faces_mediapipe(image)

            faces = [
                {"location": {"top": int(t), "right": int(r), "bottom": int(b), "left": int(l)}}
                for (t, r, b, l) in face_locations
            ]

            return {
                "success": True,
                "faces_found": len(faces),
                "faces": faces,
                "message": f"Detected {len(faces)} face(s)."
            }

        except Exception as e:
            return {"success": False, "faces_found": 0, "faces": [], "message": f"Error: {str(e)}"}

    def compare_faces(self, image_path_1: str, image_path_2: str) -> dict:
        """
        Compare two face images with multi-augmentation matching.
        Returns detailed metrics for accuracy.
        """
        try:
            img1 = cv2.imread(image_path_1)
            img2 = cv2.imread(image_path_2)

            if img1 is None:
                return {"success": False, "message": "Could not read the first image."}
            if img2 is None:
                return {"success": False, "message": "Could not read the second image."}

            locs1 = self._detect_faces_mediapipe(img1)
            locs2 = self._detect_faces_mediapipe(img2)

            if not locs1:
                return {"success": False, "message": "No face detected in the first image."}
            if not locs2:
                return {"success": False, "message": "No face detected in the second image."}

            top1, right1, bottom1, left1 = locs1[0]
            top2, right2, bottom2, left2 = locs2[0]

            face1 = img1[top1:bottom1, left1:right1]
            face2 = img2[top2:bottom2, left2:right2]

            if face1.size == 0 or face2.size == 0:
                return {"success": False, "message": "Face region too small to process."}

            # Multi-augmentation matching for robustness
            encodings1 = self._compute_augmented_encodings(face1, img1)
            encodings2 = self._compute_augmented_encodings(face2, img2)
            
            # Compute best similarity across all augmentation pairs
            all_similarities = []
            for enc1 in encodings1:
                for enc2 in encodings2:
                    sim = self._cosine_similarity(enc1, enc2)
                    all_similarities.append(sim)
            
            # Use the best match (max similarity)
            best_similarity = max(all_similarities)
            avg_similarity = sum(all_similarities) / len(all_similarities)
            
            # Final score: weighted combination (80% best, 20% average)
            final_similarity = best_similarity * 0.8 + avg_similarity * 0.2
            
            # Also compute geometric similarity separately
            geo1 = self._extract_geometric_features(img1)
            geo2 = self._extract_geometric_features(img2)
            geo_norm1 = np.linalg.norm(geo1)
            geo_norm2 = np.linalg.norm(geo2)
            
            if geo_norm1 > 0 and geo_norm2 > 0:
                geo_similarity = float(np.dot(geo1, geo2) / (geo_norm1 * geo_norm2))
            else:
                geo_similarity = 0.0
            
            is_match = final_similarity >= config.SIMILARITY_THRESHOLD
            confidence = float(f"{final_similarity * 100:.2f}")
            geo_conf = float(f"{geo_similarity * 100:.2f}")

            return {
                "success": True,
                "is_same_person": is_match,
                "confidence": confidence,
                "geometric_similarity": geo_conf,
                "feature_similarity": float(f"{best_similarity * 100:.2f}"),
                "distance": float(f"{1 - final_similarity:.4f}"),
                "message": f"{'Match!' if is_match else 'Not a match.'} Similarity: {confidence}%"
            }

        except Exception as e:
            return {"success": False, "message": f"Error: {str(e)}"}

    def list_known_faces(self) -> dict:
        """List all registered people with image counts from directory."""
        people = {}
        # Count unique names from encodings
        unique_names = set(self.known_face_names)
        
        # Also scan directory for actual image counts
        if os.path.exists(config.KNOWN_FACES_DIR):
            for person_name in os.listdir(config.KNOWN_FACES_DIR):
                person_dir = os.path.join(config.KNOWN_FACES_DIR, person_name)
                if os.path.isdir(person_dir):
                    img_count = len([f for f in os.listdir(person_dir) 
                                    if os.path.isfile(os.path.join(person_dir, f))])
                    if img_count > 0:
                        people[person_name] = img_count
                        unique_names.add(person_name)
        
        # Add entries from encodings list that don't have directories
        for name in unique_names:
            if name not in people:
                people[name] = 1

        total_images = sum(people.values())
        return {
            "success": True,
            "total_people": len(people),
            "total_faces": total_images,
            "people": [{"name": n, "samples": c} for n, c in people.items()]
        }

    def delete_face(self, name: str) -> dict:
        """Delete all face data for a person."""
        try:
            indices = [i for i, n in enumerate(self.known_face_names) if n == name]

            if not indices:
                return {"success": False, "message": f"Person '{name}' not found in database."}

            for idx in sorted(indices, reverse=True):
                self.known_face_encodings.pop(idx)
                self.known_face_names.pop(idx)

            self._save_encodings()

            person_dir = os.path.join(config.KNOWN_FACES_DIR, name)
            if os.path.exists(person_dir):
                import shutil
                shutil.rmtree(person_dir)

            return {
                "success": True,
                "message": f"Deleted {len(indices)} face sample(s) for '{name}'."
            }

        except Exception as e:
            return {"success": False, "message": f"Error: {str(e)}"}
