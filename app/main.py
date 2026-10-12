from typing import Union
from fastapi import FastAPI
from pydantic import BaseModel
from app.bigram_model import BigramModel
import spacy

import io
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from fastapi import UploadFile, File, HTTPException
from app.cnn_model import CNNModel

app = FastAPI()
nlp = spacy.load("en_core_web_md")

# Sample corpus for the bigram model
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas. "
    "It tells the story of Edmond Dantès, who is falsely imprisoned and later seeks revenge.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective"
]

bigram_model = BigramModel(corpus)


class TextGenerationRequest(BaseModel):
    start_word: str
    length: int

class EmbeddingRequest(BaseModel):
    word: str


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/generate")
def generate_text(request: TextGenerationRequest):
    generated_text = bigram_model.generate_text(
        request.start_word,
        request.length
    )
    return {"generated_text": generated_text}

@app.post("/embedding")
def get_embedding(request: EmbeddingRequest):
    vector = nlp.vocab[request.word].vector

    return {
        "word": request.word,
        "embedding": vector.tolist(),
        "dimensions": len(vector)
    }
        


# Assignment 2: CIFAR-10 CNN Image Classification

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

MODEL_PATH = Path(__file__).resolve().parent / "cnn_weights.pth"

device = torch.device("cpu")

cnn_model = CNNModel().to(device)

if MODEL_PATH.exists():
    cnn_model.load_state_dict(
        torch.load(MODEL_PATH, map_location=device, weights_only=True)
    )
    cnn_model.eval()
else:
    raise RuntimeError(f"CNN model weights not found: {MODEL_PATH}")

image_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])


@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(
            status_code=400,
            detail="Please upload a JPG or PNG image."
        )

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_tensor = image_transform(image).unsqueeze(0).to(device)

    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid image file."
        )

    with torch.no_grad():
        outputs = cnn_model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)

    return {
        "predicted_class": CIFAR10_CLASSES[predicted.item()],
        "confidence": round(confidence.item(), 4),
        "class_id": predicted.item()
    }