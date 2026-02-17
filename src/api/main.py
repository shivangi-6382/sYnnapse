import numpy as np
import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from PIL import Image
from torchvision import transforms

from src.feature_extraction.encoder import load_model, DEVICE
from src.similarity_scoring_and_retrieval.retrieval import retrieve_top_k


# ---------------- CONFIG ----------------
MODEL_PATH = "models/model.pth"
EMB_PATH = "embeddings/embeddings.npy"
LABELS_PATH = "embeddings/labels.npy"
PATHS_PATH = "embeddings/paths.npy"

DATASET_DIR = "dataset"
BASE_URL = "/static/"


# ---------------- APP INIT ----------------
app = FastAPI(title="Image Similarity Search API")

app.mount("/static", StaticFiles(directory=DATASET_DIR), name="static")


# ---------------- LOAD MODEL + DATA ----------------
model = load_model(MODEL_PATH)

embeddings = np.load(EMB_PATH)
labels = np.load(LABELS_PATH)
paths = np.load(PATHS_PATH)


# ---------------- TRANSFORM ----------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])


# ---------------- ROOT ----------------
@app.get("/")
def root():
    return {"message": "Image Similarity API is running 🚀"}


# ---------------- SEARCH ENDPOINT ----------------
@app.post("/search")
async def search_image(file: UploadFile = File(...), k: int = 5):

    # read uploaded image
    img = Image.open(file.file).convert("RGB")
    img = transform(img).unsqueeze(0).to(DEVICE)

    # generate embedding
    with torch.no_grad():
        query_emb = model(img).cpu().numpy()[0]

    # retrieve top-k
    topk_idx, topk_scores = retrieve_top_k(query_emb, embeddings, k)

    # build response
    results = []
    for idx, score in zip(topk_idx, topk_scores):
        results.append({
            "score": float(score),
            "label": str(labels[idx]),
            "image_url": BASE_URL + paths[idx]
        })

    return {"top_k_results": results}
