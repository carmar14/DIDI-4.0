import tensorflow as tf
from tensorflow.keras.models import load_model

model = load_model("model/modelo_mlp.keras")
# Convertir modelo
converter = tf.lite.TFLiteConverter.from_keras_model(model)


# (Opcional pero recomendado) cuantización
#converter.target_spec.supported_types = [tf.float16]
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

# Guardar
with open("embebido/model.tflite", "wb") as f:
    f.write(tflite_model)

# tflite_to_c_array.py

with open("embebido/model.tflite", "rb") as f:
    model_data = f.read()

with open("embebido/model.h", "w") as f:
    f.write("unsigned char model_tflite[] = {\n")

    for i, byte in enumerate(model_data):
        f.write(f"0x{byte:02x},")
        if (i + 1) % 12 == 0:
            f.write("\n")

    f.write("\n};\n")
    f.write(f"unsigned int model_tflite_len = {len(model_data)};\n")

print("Archivo model.h generado correctamente")

import numpy as np

def array_to_c(name, array):
    """Convierte numpy array a formato C"""
    flat = array.flatten()
    shape = array.shape

    c_str = f"// {name} shape {shape}\n"

    if len(shape) == 2:
        rows, cols = shape
        c_str += f"float {name}[{rows}][{cols}] = {{\n"
        for i in range(rows):
            row = ", ".join(f"{array[i][j]:.6f}" for j in range(cols))
            c_str += f"  {{{row}}},\n"
        c_str += "};\n\n"

    elif len(shape) == 1:
        c_str += f"float {name}[{shape[0]}] = {{"
        c_str += ", ".join(f"{x:.6f}" for x in array)
        c_str += "};\n\n"

    return c_str


def export_model_to_h(model, filename="embebido/weights.h"):
    weights = model.get_weights()

    with open(filename, "w") as f:
        f.write("#ifndef WEIGHTS_H\n#define WEIGHTS_H\n\n")

        layer = 1
        i = 0

        while i < len(weights):
            W = weights[i]
            b = weights[i + 1]

            f.write(array_to_c(f"W{layer}", W))
            f.write(array_to_c(f"b{layer}", b))

            i += 2
            layer += 1

        f.write("#endif\n")

    print(f"Archivo {filename} generado correctamente")


# === USO ===
export_model_to_h(model)