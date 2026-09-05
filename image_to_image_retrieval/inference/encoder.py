"""Frozen SigLIP features; model dependencies are imported only when used."""


class SigLIPEncoder:
    def __init__(self, model_name, device):
        import torch
        from transformers import AutoModel, AutoProcessor

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError("CUDA is unavailable; use --device cpu or --device auto.")
        self.device = device
        self.model = AutoModel.from_pretrained(model_name).to(device).eval()
        if self.model.config.model_type != "siglip":
            raise ValueError("--model must identify a SigLIP checkpoint compatible with this scorer.")
        self.processor = AutoProcessor.from_pretrained(model_name)

    def encode_images(self, images):
        import torch

        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        with torch.inference_mode():
            features = self.model.get_image_features(pixel_values=inputs["pixel_values"])
        return features.float().cpu()

    def encode_text(self, text):
        import torch

        # Match the notebook's padded text inputs; reject overlong prompts rather
        # than silently changing the query. The text feature stays unnormalized.
        inputs = self.processor.tokenizer(
            [text], padding="max_length", return_tensors="pt"
        ).to(self.device)
        if inputs["input_ids"].shape[1] > self.model.config.text_config.max_position_embeddings:
            raise ValueError("Query text exceeds the model's context length; use a shorter description.")
        with torch.inference_mode():
            features = self.model.get_text_features(**inputs)
        return features.float().cpu()

    def close(self):
        import torch

        del self.model
        del self.processor
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
