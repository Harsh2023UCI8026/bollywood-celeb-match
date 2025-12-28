import os
import numpy as np
from deepface import DeepFace
from joblib import dump

DATASET_PATH = "data"

embeddings = []
names = []

print("🔄 Extracting face embeddings...")

for person in os.listdir(DATASET_PATH):
    person_path = os.path.join(DATASET_PATH, person)

    if not os.path.isdir(person_path):
        continue

    for img in os.listdir(person_path):
        img_path = os.path.join(person_path, img)

        try:
            rep = DeepFace.represent(
                img_path=img_path,
                model_name="ArcFace",
                detector_backend="retinaface",
                enforce_detection=False
            )

            embeddings.append(rep[0]["embedding"])
            names.append(person)

            print(f"✅ {img_path}")

        except Exception as e:
            print(f"❌ Skipped {img_path}: {e}")

dump(embeddings, "venv/image_embeddings.joblib")
dump(names, "filenames.joblib")

print("✅ Feature extraction complete!")
