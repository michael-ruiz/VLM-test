# from transformers import BlipProcessor, BlipForConditionalGeneration
# from PIL import Image
# import torch

# # Load model + processor
# processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
# model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cuda" if torch.cuda.is_available() else "cpu")

# # Load image
# image = Image.open("test_img.png")

# # Preprocess
# inputs = processor(images=image, return_tensors="pt").to(model.device)

# # Generate caption
# out = model.generate(**inputs)
# caption = processor.decode(out[0], skip_special_tokens=True)

# print("Generated Caption:", caption)

from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import torch

# Load processor and model
processor = AutoProcessor.from_pretrained("HuggingFaceTB/SmolVLM-Instruct")
model = AutoModelForVision2Seq.from_pretrained("HuggingFaceTB/SmolVLM-Instruct").to("cuda" if torch.cuda.is_available() else "cpu")

# Load your image
image = Image.open("test_img.png")

# Prepare messages for the model (captioning prompt)
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Describe this image."}
        ]
    },
]

# Preprocess inputs
inputs = processor.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

# Generate output
outputs = model.generate(**inputs, max_new_tokens=40)
caption = processor.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
print("Generated Caption:", caption)