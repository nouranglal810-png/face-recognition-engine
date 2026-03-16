import cv2
import numpy as np
import base64
import requests

img = np.zeros((480, 640, 3), dtype=np.uint8)
# Create a dummy face so it might run
cv2.circle(img, (320, 240), 100, (255, 255, 255), -1) 

_, buffer = cv2.imencode('.jpg', img)
jpg_as_text = base64.b64encode(buffer).decode('utf-8')

res = requests.post("http://localhost:5000/api/recognize_base64", json={"image_base64": "data:image/jpeg;base64," + jpg_as_text})
print(res.json())
