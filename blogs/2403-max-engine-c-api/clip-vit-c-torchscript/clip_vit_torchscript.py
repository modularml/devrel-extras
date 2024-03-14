from pathlib import Path
from transformers import CLIPModel
import torch
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

MODEL_PATH = "models/clip_vit.torchscript"


def main():
    model_path = Path(MODEL_PATH)
    if model_path.exists():
        logger.info(f"Model exists: {MODEL_PATH}")
        return
    else:
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        model.eval()
        model.config.return_dict = False

        inputs = {
            "input_ids": torch.ones(1, 77, dtype=torch.long),
            "pixel_values": torch.rand(1, 3, 224, 224),
            "attention_mask": torch.ones((1, 77), dtype=torch.long),
        }

        with torch.no_grad():
            traced_model = torch.jit.trace(
                model, example_kwarg_inputs=dict(inputs), strict=False
            )

        traced_model.save(model_path)
        logger.info(f"Model save to {model_path}")


if __name__ == "__main__":
    main()
