import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

IMAGE_TARGET_SIZE = (224, 224)
BRACKET_NAMES = ["bracket_1_youngest", "bracket_2", "bracket_3", "bracket_4_oldest"]
BRACKET_DESCRIPTIONS = {
    "bracket_1_youngest": "younger skeletal maturity (under roughly 96 months)",
    "bracket_2": "early-middle skeletal maturity (roughly 96 to 132 months)",
    "bracket_3": "middle-later skeletal maturity (roughly 132 to 156 months)",
    "bracket_4_oldest": "older skeletal maturity (over roughly 156 months)"
}
CONFIDENCE_THRESHOLD = 0.4

st.set_page_config(page_title="Bone Age Bracket Classifier")

@st.cache_resource
def load_trained_model():
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights=None
    )
    base_model.trainable = False

    rebuilt_model = tf.keras.Sequential([
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(4, activation="softmax")
    ])

    rebuilt_model.load_weights("bone_age_classifier/model.weights.h5")
    return rebuilt_model

model = load_trained_model()

st.title("Hand X-Ray Bone Age Bracket Classifier")
st.write(
    "This tool works on hand/wrist X-rays and classifies skeletal maturity "
    "into one of four broad age brackets. It is a student coursework "
    "prototype, not a medical device, and must not be used for real "
    "clinical decisions."
)

uploaded_file = st.file_uploader("Upload a hand X-ray image (PNG)", type=["png"])

if uploaded_file is not None:
    pil_image = Image.open(uploaded_file)
    st.image(pil_image, caption="Uploaded image", width="stretch")
    grayscale_image = pil_image.convert("L")
    resized_image = grayscale_image.resize(IMAGE_TARGET_SIZE)
    image_array = np.array(resized_image)
    image_array = np.expand_dims(image_array, axis=-1)
    three_channel_array = np.repeat(image_array, 3, axis=-1)
    normalized_array = three_channel_array / 255.0
    model_input = np.expand_dims(normalized_array, axis=0)

    predictions = model.predict(model_input, verbose=0)[0]
    predicted_index = int(np.argmax(predictions))
    predicted_bracket = BRACKET_NAMES[predicted_index]
    confidence = float(predictions[predicted_index])

    st.subheader("Prediction")

    if confidence < CONFIDENCE_THRESHOLD:
        st.warning(
            f"The model is not confident about this image (highest "
            f"confidence only {confidence:.0%}). This image may look "
            f"unlike anything the model was trained on. Treat this "
            f"prediction with caution."
        )
    else:
        st.success(
            f"The model predicts this X-ray shows {BRACKET_DESCRIPTIONS[predicted_bracket]}, "
            f"with {confidence:.0%} confidence."
        )

    st.write("Full confidence breakdown:")
    for name, score in zip(BRACKET_NAMES, predictions):
        st.write(f"- {name}: {score:.1%}")
        st.divider()

with st.expander("About / How to run this app locally"):
    st.markdown("**What this app does:** Loads a trained MobileNetV2-based classifier and predicts which of four bone-age brackets a hand X-ray falls into.")
    st.markdown("**How to run locally:**")
    st.markdown("1. Clone this repository.")
    st.markdown("2. Create a virtual environment: `python3 -m venv venv` then `source venv/bin/activate`")
    st.markdown("3. Install dependencies: `pip install -r requirements.txt`")
    st.markdown("4. Run the app: `streamlit run app.py`")
    st.markdown("5. Open the local URL shown in the terminal (usually http://localhost:8501)")
    st.markdown("**Note:** This is a coursework prototype, not a medical device.")