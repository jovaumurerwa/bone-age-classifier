import numpy as np
import streamlit as st

from bone_age.integrity import ModelIntegrityError
from bone_age.model import BRACKET_NAMES, build_and_load_model
from bone_age.preprocess import InvalidImageError, load_safe_image, to_model_input

BRACKET_DESCRIPTIONS = {
    "bracket_1_youngest": "younger skeletal maturity (under roughly 96 months)",
    "bracket_2": "early-middle skeletal maturity (roughly 96 to 132 months)",
    "bracket_3": "middle-later skeletal maturity (roughly 132 to 156 months)",
    "bracket_4_oldest": "older skeletal maturity (over roughly 156 months)",
}
CONFIDENCE_THRESHOLD = 0.4

st.set_page_config(page_title="Bone Age Bracket Classifier")


@st.cache_resource
def get_model():
    return build_and_load_model()


st.title("Hand X-Ray Bone Age Bracket Classifier")
st.write(
    "Classifies a hand/wrist X-ray into one of four skeletal-maturity brackets. "
    "This is a student prototype, not a medical device, and must not be used for clinical decisions."
)
st.caption("Privacy: images are processed in memory only. Nothing is saved or logged.")

try:
    model = get_model()
except ModelIntegrityError:
    st.error("The model file failed its integrity check, so the app will not run.")
    st.stop()

uploaded_file = st.file_uploader("Upload a hand X-ray (PNG or JPEG, max 5 MB)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    try:
        image = load_safe_image(uploaded_file.getvalue())
    except InvalidImageError as error:
        st.error(str(error))
        st.stop()

    st.image(image, caption="Uploaded image", width="stretch")

    predictions = model.predict(to_model_input(image), verbose=0)[0]
    predicted_index = int(np.argmax(predictions))
    predicted_bracket = BRACKET_NAMES[predicted_index]
    confidence = float(predictions[predicted_index])

    st.subheader("Prediction")
    if confidence < CONFIDENCE_THRESHOLD:
        st.warning(
            f"The model is not confident about this image (highest score only {confidence:.0%}). "
            "It may not look like the X-rays the model was trained on."
        )
    else:
        st.success(f"Predicted: {BRACKET_DESCRIPTIONS[predicted_bracket]}, with {confidence:.0%} confidence.")

    st.write("Full confidence breakdown:")
    for name, score in zip(BRACKET_NAMES, predictions):
        st.write(f"- {name}: {score:.1%}")
    st.divider()
