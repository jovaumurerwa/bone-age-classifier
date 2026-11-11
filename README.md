# Bone Age Bracket Classifier

A deep learning app that estimates **skeletal maturity from a hand X-ray** and places it in one of four age brackets. Built with transfer learning (MobileNetV2) and served through a Streamlit web app, with security controls for handling medical images.

> ⚠️ Student project. Not a medical device. Must not be used for clinical decisions.

## The problem

Doctors estimate a child's "bone age" from hand X-rays to spot growth disorders. Doing it by hand is slow and depends on the reader's experience. In places with few pediatric radiologists, including many parts of Africa, a quick first estimate can help prioritise which cases a specialist should look at first.

## How it works

1. **Upload** a hand/wrist X-ray (PNG or JPEG).
2. The image is **checked** (type, size, dimensions) before it is decoded.
3. It is converted to grayscale, resized to 224×224 and scaled to 0–1.
4. A **MobileNetV2** model (pretrained base, custom head) predicts one of 4 brackets.
5. The app shows the bracket, the confidence, and a warning when confidence is low.

| Bracket | Approximate age |
|---|---|
| 1 | under 96 months |
| 2 | 96–132 months |
| 3 | 132–156 months |
| 4 | over 156 months |

## Results

<!-- TODO: fill in with your real numbers from the training notebook, then delete this line -->
| Metric | Value |
|---|---|
| Validation accuracy | TODO |
| Macro F1 | TODO |

<!-- TODO: add your confusion matrix image: docs/confusion_matrix.png -->

## Security and privacy

Medical images are sensitive, so the app is built to handle them carefully:

| Risk | Protection |
|---|---|
| Malicious or fake files | Only real PNG/JPEG images pass. Files are verified by their content, not their name |
| Decompression bombs (tiny files that expand to huge images) | Pixel limit checked **before** decoding |
| Oversized uploads | 5 MB limit in the app and in Streamlit config |
| Tampered model file | SHA-256 fingerprint checked before loading; the app refuses to run on mismatch |
| Code hidden inside model files | Architecture is rebuilt in code and only numeric weights are loaded |
| Patient data leaks | Images stay in memory only; nothing is saved or logged; usage stats are off |
| Leaking error details | Stack traces hidden from users |
| Vulnerable libraries | `pip-audit` runs in CI on every push |
| Unsafe code / leaked secrets | Bandit and Gitleaks run in CI on every push |

## Tech stack

Python · TensorFlow / Keras · MobileNetV2 · Streamlit · Pillow · NumPy · Pytest · GitHub Actions

## Run it locally

```bash
git clone https://github.com/jovaumurerwa/bone-age-classifier.git
cd bone-age-classifier
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501.

## Run the tests

```bash
pip install -r requirements-dev.txt
pytest -v
```

The tests check image validation (empty files, non-images, wrong formats, oversized and tiny images, decompression bombs) and the model integrity check.

## Project structure

```
app.py                      Streamlit interface
bone_age/preprocess.py      Upload validation and image preparation
bone_age/integrity.py       SHA-256 check of the model weights
bone_age/model.py           Model architecture and weight loading
bone_age_classifier/        Trained weights
tests/                      Pytest suite
.streamlit/config.toml      Upload limit and privacy settings
```

## Limitations

- Four broad brackets, not an exact age in months.
- Trained on one public dataset; performance on X-rays from other machines or populations is unknown.
- Sex is a strong factor in bone age and is not used as an input yet.

## What I would improve next

- Predict age in months (regression) and add sex as an input.
- Test on X-rays from a different source to check the model generalises.
- Add Grad-CAM heatmaps so users can see which parts of the X-ray drove the prediction.

## What I learned

- Transfer learning lets a small dataset reach useful accuracy.
- An ML app is also a web app: uploads, model files and dependencies all need security.
- Showing confidence and limits honestly matters more in health than a single "answer".
