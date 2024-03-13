import argparse
import logging
import requests
from pathlib import Path
from requests.exceptions import RequestException
from PIL import Image, UnidentifiedImageError
from transformers import CLIPProcessor
import numpy as np

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(description="Resnet build example")
aa = parser.add_argument
aa(
    "--text",
    type=str,
    help="inputs comma separated text for text-image similarity based on clip-vit",
)
aa("--url", type=str, help="url to download the image")
args = parser.parse_args()


def download_image(url):
    try:
        response = requests.get(url, stream=True).raw
        image = Image.open(response)
        return image
    except RequestException as e:
        logger.exception(f"Request failed: {e}")
    except UnidentifiedImageError:
        logger.exception("The downloaded content is not a valid image.")
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")


def save_as_bin(inputs):
    input_dir = Path("inputs")
    input_dir.mkdir(exist_ok=True)
    created_files = []
    for name, value in inputs.items():
        if name in ("input_ids", "attention_mask"):
            value = value.astype(np.int64)
        elif name == "pixel_values":
            value = value.astype(np.float32)
        else:
            raise ValueError(f"Unknown input name for clip-vit model. Given {name}")

        filename = input_dir / name
        filename = filename.with_suffix(".bin")
        filename.unlink(missing_ok=True)
        value.tofile(filename)

        shape = np.array(value.shape).astype(np.int64)
        shape_file = input_dir / f"{name}_shape.bin"
        shape.tofile(shape_file)
        created_files += [str(filename), str(shape_file)]

    logger.info(f"Created files: {created_files}")
    return


def main():
    image = download_image(args.url)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    inputs = processor(
        text=args.text.split(","), images=image, return_tensors="np", padding=True
    )
    save_as_bin(inputs)
    return


if __name__ == "__main__":
    main()
