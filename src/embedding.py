import torch
import open_clip
from PIL import Image
import torch.nn.functional as F

class CLIPEmbedder:
    def __init__(self, model_name="ViT-B-32", pretrained="openai", device=None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Loading CLIP model {model_name} on {self.device}...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, 
            pretrained=pretrained,
            device=self.device
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()

    def get_visual_embedding(self, image: Image.Image) -> torch.Tensor:
        """
        Embed the image using the frozen/fine-tuned vision encoder.
        """
        image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            vis_emb = self.model.encode_image(image_tensor)
        return F.normalize(vis_emb, dim=-1)

    def get_text_embedding(self, text: str) -> torch.Tensor:
        """
        Embed the caption text using the frozen text encoder.
        """
        text_tokens = self.tokenizer([text]).to(self.device)
        with torch.no_grad():
            text_emb = self.model.encode_text(text_tokens)
        return F.normalize(text_emb, dim=-1)

    def compute_fusion_embedding(self, image: Image.Image, caption: str, alpha: float) -> torch.Tensor:
        """
        Compute the alpha-weighted fused embedding of vision and text:
        v_i = alpha * v_vis + (1 - alpha) * v_text
        Returned vector is L2-normalized.
        """
        v_vis = self.get_visual_embedding(image)
        v_text = self.get_text_embedding(caption)
        
        fused = (alpha * v_vis) + ((1.0 - alpha) * v_text)
        return F.normalize(fused, dim=-1)
