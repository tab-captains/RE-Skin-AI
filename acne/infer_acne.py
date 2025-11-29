from pathlib import Path

import torch
from torchvision import transforms
from PIL import Image

from .models.resnet50_acne import build_resnet50_acne


class AcnePredictor:
    def __init__(self, checkpoint_path, device):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)

        ckpt_path = Path(checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location=self.device)

        class_names = ckpt["class_names"]
        self.class_names = class_names

        self.model = build_resnet50_acne(num_classes=len(class_names))
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        # val용이랑 같은 transform
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def _prepare(self, img: Image.Image) -> torch.Tensor:
        x = self.transform(img).unsqueeze(0)  # (1, C, H, W)
        return x.to(self.device)

    @torch.no_grad()
    def predict_pil(self, img: Image.Image) -> dict:
        x = self._prepare(img)
        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().tolist()

        max_idx = int(torch.argmax(torch.tensor(probs)).item())
        pred_class = self.class_names[max_idx]

        return {
            "pred_class": pred_class,
            "probs": {
                cls: float(p) for cls, p in zip(self.class_names, probs)
            },
        }

    @torch.no_grad()
    def predict_path(self, img_path: str | Path) -> dict:
        img = Image.open(img_path).convert("RGB")
        return self.predict_pil(img)
