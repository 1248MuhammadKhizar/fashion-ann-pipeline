import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow import keras

d = np.load("data/processed/data.npz")
model = keras.models.load_model("models/model.h5")
loss, acc = model.evaluate(d["x_test"], d["y_test"], verbose=0)

pred = np.argmax(model.predict(d["x_test"], verbose=0), axis=1)
ConfusionMatrixDisplay(confusion_matrix(d["y_test"], pred)).plot(cmap="Blues")
plt.savefig("confusion_matrix.png")

json.dump({"test_loss": float(loss), "test_accuracy": float(acc)}, open("metrics.json", "w"), indent=2)
print(f"loss={loss:.4f} acc={acc:.4f}")
