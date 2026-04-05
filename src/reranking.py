import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
from typing import List, Dict

class BLIP2Reranker:
    def __init__(self, model_name="Salesforce/blip2-opt-2.7b", device=None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Loading BLIP-2 model {model_name} for ITM reranking on {self.device}...")
        dtype = torch.float16 if self.device != "cpu" else torch.float32
        
        self.processor = Blip2Processor.from_pretrained(model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name, 
            torch_dtype=dtype
        ).to(self.device)
        self.model.eval()

    def get_itm_score(self, image: Image.Image, text: str) -> float:
        """
        Calculates a pseudo-ITM score using the language modeling loss 
        of the text given the image. Lower loss means higher likelihood/match.
        Returns negative loss so that higher score = better match.
        """
        inputs = self.processor(image, text=text, return_tensors="pt").to(self.device, self.model.dtype)
        # For conditional generation, input text is passed as labels to compute loss
        labels = inputs.input_ids.clone()
        # Optionally, ignore padding token by setting to -100
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        
        with torch.no_grad():
            # For BLIP-2 generative models in huggingface, provide labels
            # Ensure pixel_values are dtype compatible
            outputs = self.model(
                pixel_values=inputs.pixel_values, 
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                labels=labels
            )
            
        loss = outputs.loss.item()
        return -loss

    def rerank(self, query_image: Image.Image, candidates: List[Dict]) -> List[Dict]:
        """
        Given a query image and a list of candidate dictionaries containing 'caption',
        computes the ITM score and sorts candidates descending by score.
        """
        scored_candidates = []
        for cand in candidates:
            score = self.get_itm_score(query_image, cand['caption'])
            new_cand = cand.copy()
            new_cand['itm_score'] = score
            scored_candidates.append(new_cand)
            
        # Re-rank: Highest score first
        ranked = sorted(scored_candidates, key=lambda x: x['itm_score'], reverse=True)
        return ranked
