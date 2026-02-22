from transformers import BlipProcessor, BlipForQuestionAnswering
from PIL import Image
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Usando:", device)

processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base").to(device)

image = Image.open("ejemplo.jpeg").convert("RGB")

question = "How many people are in the image?"

inputs = processor(image, question, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}
with torch.no_grad():
    out = model.generate(**inputs)
answer = processor.decode(out[0], skip_special_tokens=True)

print("Respuesta:", answer)
