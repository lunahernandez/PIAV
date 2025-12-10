# https://huggingface.co/Intel/dpt-hybrid-midas

import os
from PIL import Image
import numpy as np
import torch
from transformers import DPTImageProcessor, DPTForDepthEstimation

model_id = "Intel/dpt-hybrid-midas"
mi_token = os.getenv("HF_TOKEN")

if mi_token is None:
    raise ValueError("Error: No se encontró la variable de entorno 'HF_TOKEN'.")

print("Cargando modelo...")
image_processor = DPTImageProcessor.from_pretrained(model_id, token=mi_token)
model = DPTForDepthEstimation.from_pretrained(model_id, low_cpu_mem_usage=True, token=mi_token)

ruta_imagen = "images/im0.png"

try:
    image = Image.open(ruta_imagen)
except FileNotFoundError:
    print(f"Error: No se encontró el archivo en: {ruta_imagen}")
    exit()

print("Procesando imagen...")

inputs = image_processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    predicted_depth = outputs.predicted_depth

prediction = torch.nn.functional.interpolate(
    predicted_depth.unsqueeze(1),
    size=image.size[::-1],
    mode="bicubic",
    align_corners=False,
)

output = prediction.squeeze().cpu().numpy()
formatted = (output * 255 / np.max(output)).astype("uint8")
depth = Image.fromarray(formatted)
depth.show()
depth.save(f"output/depth_{os.path.basename(ruta_imagen)}")
print("¡Listo! Imagen guardada.")