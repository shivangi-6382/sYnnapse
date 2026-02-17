import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ImageEncoder(nn.Module):
    def __init__(self, emb_dim=512):
        super().__init__()

        backbone = models.resnet50(weights="IMAGENET1K_V1")
        self.features = nn.Sequential(*list(backbone.children())[:-1])

        # freeze backbone
        for p in self.features.parameters():
            p.requires_grad = False

        self.embedding = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, emb_dim),
            nn.ReLU(),
            nn.BatchNorm1d(emb_dim)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.embedding(x)
        return torch.nn.functional.normalize(x, dim=1)


# ---------- image preprocessing ----------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


def preprocess_image(img_path):
    img = Image.open(img_path).convert("RGB")
    img = transform(img).unsqueeze(0)  # add batch dim
    return img.to(DEVICE)


# ---------- load trained model ----------
def load_model(model_path):
    model = ImageEncoder()

    ckpt = torch.load(model_path, map_location=DEVICE)

    # handle both saved formats
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)

    model.to(DEVICE)
    model.eval()
    return model


# ---------- get embedding ----------
def get_embedding(model, img_path):
    img = preprocess_image(img_path)

    with torch.no_grad():
        emb = model(img)

    return emb.cpu().numpy()
