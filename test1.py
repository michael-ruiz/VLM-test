from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# Load model + processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cuda" if torch.cuda.is_available() else "cpu")

# Load image
image = Image.open("test_img.png")

# Preprocess
inputs = processor(images=image, return_tensors="pt").to(model.device)

# Generate caption
out = model.generate(**inputs)
caption = processor.decode(out[0], skip_special_tokens=True)

print("Generated Caption:", caption)
