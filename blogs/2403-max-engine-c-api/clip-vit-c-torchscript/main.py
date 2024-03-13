import requests
from PIL import Image
from max import engine
import torch
from transformers import CLIPProcessor

url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
inputs = processor(
    text=["a photo of a cat", "a photo of a dog"],
    images=image,
    return_tensors="pt",
    padding=True,
)
print(
    inputs["input_ids"].size(),
    inputs["attention_mask"].size(),
    inputs["pixel_values"].size(),
)
print(
    inputs["input_ids"].type(),
    inputs["attention_mask"].type(),
    inputs["pixel_values"].type(),
)

session = engine.InferenceSession()
input_spec_lst = [
    engine.TorchInputSpec(shape=inputs["input_ids"].size(), dtype=engine.DType.int64),
    engine.TorchInputSpec(
        shape=inputs["pixel_values"].size(), dtype=engine.DType.float32
    ),
    engine.TorchInputSpec(
        shape=inputs["attention_mask"].size(), dtype=engine.DType.int64
    ),
]
print(input_spec_lst)
options = engine.TorchLoadOptions(input_spec_lst)
clip_vit = session.load("models/clip_vit.torchscript", options)
outputs = clip_vit.execute(**inputs)

logits = torch.from_numpy(outputs["result0"])
scores = logits.softmax(dim=-1)
print(scores.numpy())
