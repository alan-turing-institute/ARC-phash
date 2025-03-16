import torch
from huggingface_hub import hf_hub_download
from timm import create_model
from torchvision import transforms
from transformers import pipeline

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Custom image classification pipeline
class CustomImageClassificationPipeline:
    def __init__(self, model):
        self.model = model
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def post_process(self, result):
        softmax_scores = torch.softmax(result.detach(), dim=1).cpu().numpy()[0]
        return [
            {"label": "artificial", "score": softmax_scores[0].item()},
            {"label": "human", "score": softmax_scores[1].item()},
        ]

    def __call__(self, image):
        image = self.transform(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            return self.post_process(self.model(image))


# Load model directly
def load_models(model_str):
    if model_str == "Dafilab/ai-image-detector":
        # Download model from HuggingFace Hub
        MODEL_PATH = hf_hub_download(
            repo_id="Dafilab/ai-image-detector", filename="pytorch_model.pth"
        )
        # Load model
        model: torch.nn.Module = create_model(
            "efficientnet_b4", pretrained=False, num_classes=2
        )
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        return CustomImageClassificationPipeline(model)

    if model_str == "Organika/sdxl-detector":
        return pipeline(
            "image-classification",
            model="Organika/sdxl-detector",
            device_map=DEVICE,
        )

    error_str = f"Unknown Model: {model_str}"
    raise ValueError(error_str)


class AIDetector:
    def __init__(self, model_str):
        self.model = load_models(model_str)

    def post_process(self, result):
        return result[0]["score"] >= 0.5

    def detect(self, image):
        return self.model(image)

    def __call__(self, image):
        return self.post_process(self.detect(image))
