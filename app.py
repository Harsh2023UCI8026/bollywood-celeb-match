import streamlit as st
from deepface import DeepFace
from PIL import Image
import numpy as np
import tempfile
import os

st.set_page_config(
    page_title="Celebrity Look-Alike Finder",
    page_icon="🎭",
    layout="centered"
)

st.title("🎭 Celebrity Look-Alike Finder")
st.write("Upload your photo and find which celebrity you resemble!")

uploaded_file = st.file_uploader(
    "Upload your photo",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Find My Look-Alike"):
        with st.spinner("Analyzing your face... 🔍"):
            try:
                # Save image temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
                    image.save(temp.name)
                    img_path = temp.name

                # Face recognition
                result = DeepFace.find(
                    img_path=img_path,
                    db_path="data",   # folder must exist
                    enforce_detection=False,
                    model_name="VGG-Face"
                )

                st.success("🎉 Match Found!")
                st.write(result)

            except Exception as e:
                st.error("❌ Something went wrong")
                st.code(str(e))



