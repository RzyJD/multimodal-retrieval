# Third-party sources and permissions

This project uses pretrained models and code from the sources below. Project-specific adaptations do not imply authorship of those upstream methods. No blanket repository license supersedes upstream terms.

| Component | Source and relationship | Permission status |
| --- | --- | --- |
| `tip_adapter_ft/main_imagenet.py`, `imagenet.py`, `utils.py` and adaptation structure | Derived from [Tip-Adapter](https://github.com/gaopengcuhk/Tip-Adapter), with task-specific prompts, F1 evaluation and dataset handling | An explicit license for the copied Tip-Adapter source was not found during review. Redistribution permission remains to be confirmed; citation alone is not permission. |
| `tip_adapter_ft/clip/` | Bundled code derived from [OpenAI CLIP](https://github.com/openai/CLIP) | MIT. The upstream notice is retained in `tip_adapter_ft/clip/LICENSE`. This does not license the entire adaptation directory. |
| Taiyi-CLIP | [IDEA-CCNL model card](https://huggingface.co/IDEA-CCNL/Taiyi-CLIP-Roberta-large-326M-Chinese) | Downloaded separately; consult the model card and upstream terms. |
| SigLIP | [Google model card](https://huggingface.co/google/siglip-base-patch16-224) | Downloaded separately; consult the model card and upstream terms. |
| LLaVA | [Source](https://github.com/haotian-liu/LLaVA), [v1.5-7B model card](https://huggingface.co/liuhaotian/llava-v1.5-7b) | Installed separately. Source and model terms are distinct; the model card identifies the Llama 2 Community License. |

No datasets or model checkpoints are redistributed. Notebook media outputs are excluded. The homepage framework figure includes illustrative thumbnails from the original project figure; these do not grant rights to the underlying photographs. Their original redistribution terms have not been verified. The ignored `tip_adapter_ft/exp.log` and `cache_model.png` are upstream reference artifacts, not original project results.

## References

- Radford et al. (2021), *Learning Transferable Visual Models From Natural Language Supervision* — CLIP.
- Zhang et al. (2022), *Tip-Adapter: Training-free Adaption of CLIP for Few-shot Classification* — Tip-Adapter and Tip-Adapter-F.
- Zhai et al. (2023), *Sigmoid Loss for Language Image Pre-Training* — SigLIP.
- Liu et al. (2023, 2024), *Visual Instruction Tuning* and *Improved Baselines with Visual Instruction Tuning* — LLaVA.

Full research references appear in the project report.
