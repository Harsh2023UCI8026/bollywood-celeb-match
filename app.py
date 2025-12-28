import streamlit as st
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
from joblib import load
from PIL import Image
import numpy as np

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Celebrity Look-Alike",
    page_icon="🎭",
    layout="centered"
)

# ---------- LOAD DATA ----------
embeddings = load("venv/image_embeddings.joblib")
names = load("filenames.joblib")

# ---------- HEADER ----------
st.markdown(
    """
    <h1 style='text-align:center;'> Which Bollywood Celebrity Do You Look Like?</h1>
    <p style='text-align:center; color:gray;'>Made by harsh jha</p>
    <hr>
    """,
    unsafe_allow_html=True
)

# ---------- IMAGE UPLOAD ----------
uploaded_image = st.file_uploader(
    "Upload your image (JPG / PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_image:

    col1, col2 = st.columns([1, 1.2])

    with col1:
        image = Image.open(uploaded_image).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

    # Save temporary image
    temp_path = "temp.jpg"
    image.save(temp_path)

    with st.spinner(" Analyzing face..."):
        embedding = DeepFace.represent(
            img_path=temp_path,
            model_name="ArcFace",
            detector_backend="retinaface",
            enforce_detection=False
        )[0]["embedding"]

        similarities = cosine_similarity([embedding], embeddings)[0]
        best_index = np.argmax(similarities)
        confidence = round(similarities[best_index] * 100, 2)

    with col2:
        st.markdown("###  Result")
        st.success(f"**This image looks like:** {names[best_index]}")
        st.metric(label="Similarity Score", value=f"{confidence}%")

    st.progress(min(int(confidence), 100))

    st.markdown("---")
    st.caption("⚡ Powered by DeepFace + Streamlit | Model: ArcFace")















