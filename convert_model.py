from tensorflow.keras.models import load_model

model = load_model("dermalscan_tf_compatible", compile=False)
model.save("dermalscan_final.keras")   # or .h5
print("Model converted successfully")
