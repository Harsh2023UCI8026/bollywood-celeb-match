import streamlit as st
import numpy as np
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
from joblib import load

# ---------- CONFIG ----------
st.set_page_config(
    page_title="Celebrity Look-Alike",
    page_icon="🎭",
    layout="centered"
)

# ---------- LOAD EMBEDDINGS ----------
@st.cache_resource
def load_embeddings():
    embeddings, names = load("image_embeddings.joblib")
    return embeddings, names

embeddings, names = load_embeddings()

# ---------- UI ----------
st.title("🎭 Celebrity Look-Alike Finder")

uploaded = st.file_uploader("Upload your photo", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing face..."):
        temp_path = "temp.jpg"
        image.save(temp_path)

        result = DeepFace.represent(
            img_path=temp_path,
            model_name="ArcFace",
            detector_backend="retinaface",
            enforce_detection=False
        )

        query_embedding = result[0]["embedding"]
        similarities = cosine_similarity([query_embedding], embeddings)[0]

        best_idx = int(np.argmax(similarities))
        confidence = round(float(similarities[best_idx]) * 100, 2)

    st.success(f"🎯 Match: **{names[best_idx]}**")
    st.metric("Confidence", f"{confidence}%")

















