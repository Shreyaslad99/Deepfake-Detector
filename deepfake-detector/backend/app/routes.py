from flask import Blueprint, request, render_template
import os
from .models import predict_image
from .video_processor import predict_video

main = Blueprint("main", __name__)

UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp", "bmp"}
VIDEO_EXTENSIONS = {'.mp4', '.mov', '.avi', '.mkv'}

@main.route("/")
def home():
    return render_template("index.html")

@main.route("/predict-image", methods=["POST"])
def predict_media():
    if "file" not in request.files:
        return {"error": "No file uploaded"}, 400

    file = request.files["file"]
    
    if file.filename == '':
        return {"error": "No selected file"}, 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    try:
        _, ext = os.path.splitext(filepath.lower())

        if ext in VIDEO_EXTENSIONS:
            combined_fake_prob, peak_fake_prob = predict_video(filepath)
            breakdown_text = f"Temporal Analysis (Peak Frame: {round(peak_fake_prob * 100, 2)}% AI)"
            final_label = "AI VIDEO" if combined_fake_prob >= 0.55 else "REAL VIDEO"
            threshold = 0.55
        else:
            combined_fake_prob = predict_image(filepath)
            breakdown_text = "Vision Transformer Model"
            final_label = "AI IMAGE" if combined_fake_prob >= 0.60 else "REAL IMAGE"
            threshold = 0.60

        final_conf = combined_fake_prob if combined_fake_prob >= threshold else 1.0 - combined_fake_prob

        return {
            "result": final_label,
            "overall_confidence": round(final_conf * 100, 2),
            "breakdown": {
                "transformer_model_prob": round(combined_fake_prob * 100, 2),
                "method": breakdown_text
            }
        }

    except Exception as e:
        return {"error": str(e)}, 500