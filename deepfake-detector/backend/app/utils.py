import cv2
import numpy as np
from PIL import Image

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def extract_face(file_path):
    image = Image.open(file_path).convert("RGB")
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) > 0:
        x, y, w, h = faces[0]
        face = img_array[y:y+h, x:x+w]
        return Image.fromarray(face)
    
    return image

def check_artifacts(file_path):
    image = cv2.imread(file_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    fake_prob = max(0.0, min(1.0, 1.0 - (variance / 500.0)))
    return fake_prob