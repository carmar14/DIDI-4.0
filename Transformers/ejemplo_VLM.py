from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image
import torch

processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")

image = Image.open("ejemplo.jpeg").convert("RGB")

question = "How many people are in the image?"

inputs = processor(image, question, return_tensors="pt")

out = model.generate(**inputs)
answer = processor.decode(out[0], skip_special_tokens=True)

print("Respuesta:", answer)
