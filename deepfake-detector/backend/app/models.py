import torch
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image

# 97.5% accurate model — labels: {0: "Artificial", 1: "Deepfake", 2: "Real"}
MODEL_NAME = "prithivMLmods/AI-vs-Deepfake-vs-Real"

processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
model = ViTForImageClassification.from_pretrained(MODEL_NAME)
model.eval()

# Hardcoded label indices — no string matching needed
AI_LABEL_INDICES = {0, 1}  # 0=Artificial, 1=Deepfake are both "not real"
REAL_LABEL_INDEX = 2        # 2=Real

def predict_image(image_input):
    """Analyzes a single image or PIL frame. Returns AI/fake probability 0.0–1.0."""
    if isinstance(image_input, str):
        image = Image.open(image_input).convert("RGB")
    else:
        image = image_input

    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)[0]

    # Combine Artificial + Deepfake probabilities as the "fake" score
    fake_prob = probs[0].item() + probs[1].item()  # Artificial + Deepfake
    fake_prob = min(fake_prob, 1.0)

    return fake_prob
