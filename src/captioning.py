import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image

class BLIP2Captioner:
    def __init__(self, model_name="Salesforce/blip2-opt-2.7b", device=None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Loading BLIP-2 model {model_name} on {self.device}...")
        # To avoid massive memory usage, we can opt to load in fp16 if possible
        dtype = torch.float16 if self.device != "cpu" else torch.float32
        
        self.processor = Blip2Processor.from_pretrained(model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name, 
            torch_dtype=dtype
        ).to(self.device)
        self.model.eval() # Keep frozen

    def generate_caption(self, image: Image.Image, prompt="A photo of a clothing item showing color, fit, material, and style. It is ", gt_caption=None) -> str:
        # Generate a caption for the given PIL image.
        if gt_caption is not None:
            return gt_caption
            
        # BLIP-2 can be prompted.
        inputs = self.processor(image, text=prompt, return_tensors="pt").to(self.device, self.model.dtype)
        
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=50)
            
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        # Combine prompt context and generated text
        if prompt:
            full_text = prompt + generated_text
        else:
            full_text = generated_text
            
        return full_text
