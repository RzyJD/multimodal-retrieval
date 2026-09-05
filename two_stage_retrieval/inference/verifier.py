"""LLaVA v1.5 candidate verification, based on the project's VQA notebook."""

import gc
import re

from .retrievers import checked_device


def parse_answer(answer):
    """Accept an unambiguous leading Yes/No; expose uncertainty separately."""
    words = set(re.findall(r"\b(?:yes|no)\b", answer, flags=re.IGNORECASE))
    words = {word.lower() for word in words}
    match = re.match(r"^\s*(yes|no)\b", answer, flags=re.IGNORECASE)
    if not match or len(words) != 1:
        return None
    return match.group(1).lower() == "yes"


class LLaVAVerifier:
    def __init__(self, config):
        from llava.model.builder import load_pretrained_model
        from llava.utils import disable_torch_init

        self.device = checked_device(config.device)
        if self.device.type != "cuda":
            raise ValueError("This LLaVA v1.5 verifier requires CUDA.")
        disable_torch_init()
        # This entry point supports the v1.5 LLaMA/Vicuna model family. Naming
        # it explicitly also supports a local checkpoint folder with any name.
        self.tokenizer, self.model, self.processor, _ = load_pretrained_model(
            config.llava_model, model_base=None, model_name="llava-v1.5-7b",
            device=str(self.device), device_map=str(self.device),
        )
        self.model.eval()

    def verify(self, image_path, query):
        import torch
        from PIL import Image
        from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
        from llava.conversation import conv_templates
        from llava.mm_utils import process_images, tokenizer_image_token

        with Image.open(image_path) as source:
            image = source.convert("RGB")
        try:
            tensors = process_images([image], self.processor, self.model.config)
            if isinstance(tensors, list):
                tensors = [tensor.to(self.model.device, dtype=torch.float16) for tensor in tensors]
            else:
                tensors = tensors.to(self.model.device, dtype=torch.float16)
            image_token = DEFAULT_IMAGE_TOKEN
            if getattr(self.model.config, "mm_use_im_start_end", False):
                image_token = f"{DEFAULT_IM_START_TOKEN}{image_token}{DEFAULT_IM_END_TOKEN}"
            conversation = conv_templates["llava_v1"].copy()
            prompt = f"Is this an image of a {query}? Please answer with Yes or No."
            conversation.append_message(conversation.roles[0], f"{image_token}\n{prompt}")
            conversation.append_message(conversation.roles[1], None)
            inputs = tokenizer_image_token(
                conversation.get_prompt(), self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt",
            ).unsqueeze(0).to(self.model.device)
            with torch.inference_mode():
                generated = self.model.generate(
                    inputs, images=tensors, image_sizes=[image.size],
                    do_sample=False, max_new_tokens=32, use_cache=True,
                )
            # Support versions returning either continuation-only or full IDs.
            if generated.shape[1] >= inputs.shape[1] and torch.equal(generated[:, :inputs.shape[1]], inputs):
                generated = generated[:, inputs.shape[1]:]
            return self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
        finally:
            image.close()

    def close(self):
        import torch

        self.model = None
        self.tokenizer = None
        self.processor = None
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
