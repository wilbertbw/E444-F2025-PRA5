import os
import logging
import threading
from typing import Optional
from flask import Flask, request, jsonify, render_template_string

# Flask app (Elastic Beanstalk Procfile expects "application:application")
application = Flask(__name__)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Resolve artifact paths relative to this file; allow env overrides (empty env won't override)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.getenv("MODEL_PATH") or os.path.join(BASE_DIR, "basic_classifier.pkl")
VECTORIZER_PATH = os.getenv("VECTORIZER_PATH") or os.path.join(BASE_DIR, "count_vectorizer.pkl")

# Log resolved paths
logger.info("CWD: %s", os.getcwd())
logger.info("Resolved MODEL_PATH: %s", MODEL_PATH)
logger.info("Resolved VECTORIZER_PATH: %s", VECTORIZER_PATH)

# Global variables for loaded artifacts
_loaded_model: Optional[object] = None
_vectorizer: Optional[object] = None
_artifact_lock = threading.Lock()

# Artifact loading
def _load_artifacts_once() -> None:
    """Lazily load model and vectorizer once per process."""
    global _loaded_model, _vectorizer
    if _loaded_model is not None and _vectorizer is not None:
        return
    with _artifact_lock:
        if _loaded_model is None or _vectorizer is None:
            import pickle
            logger.info("Loading artifacts...")
            with open(MODEL_PATH, "rb") as mf:
                _loaded_model = pickle.load(mf)
            with open(VECTORIZER_PATH, "rb") as vf:
                _vectorizer = pickle.load(vf)
            logger.info("Artifacts loaded.")

# Inference function
def _predict_text(message: str) -> str:
    """Run inference and return the predicted class as a string label."""
    _load_artifacts_once()
    X = _vectorizer.transform([message])
    pred = _loaded_model.predict(X)
    # pred[0] could be a numpy scalar; normalize to native str
    val = pred[0]
    val_py = val.item() if hasattr(val, "item") else val
    return str(val_py)

# Eager load artifacts in a background thread at startup
def _eager_load_background():
    try:
        _load_artifacts_once()
    except Exception as e:
        # Log and continue; app remains healthy and will lazy-load on first request
        logger.warning("Background eager load failed: %s", e, exc_info=True)


# Non-blocking eager load at startup
threading.Thread(target=_eager_load_background, daemon=True).start()

DEMO_HTML = """
<!doctype html>
<html lang="en">
  <head>
  </head>
  <body>
    <h1>Text Classifier Demo</h1>
    <p>Enter text to classify:</p>
    <p id="errorMsg"></p>
    <textarea id="message" placeholder="Type here..."></textarea>
    <button id="predict-btn">Predict</button>
    <p id="prediction"></p>
    <script>
    document.getElementById('predict-btn').onclick = async function() {
        const message = document.getElementById('message').value.trim();
        if (!message) {
            document.getElementById('errorMsg').textContent = "Please enter a message.";
            return;
        }
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({ message })
        });
        const data = await response.json();
        if (data.label !== undefined) {
            document.getElementById('prediction').textContent = "Prediction: " + data.label;
        } else {
            document.getElementById('message').textContent = (data.error || "Unknown error");
        }
    };
    </script>
  </body>
</html>
"""

# Routes
@application.get("/")
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": bool(_loaded_model is not None and _vectorizer is not None),
        "model_path": MODEL_PATH,
        "vectorizer_path": VECTORIZER_PATH,
    }), 200

# Demo page rendering endpoint
@application.get("/demo")
def demo():
    return render_template_string(
        DEMO_HTML,
        model_loaded=bool(_loaded_model is not None and _vectorizer is not None),
        model_path=MODEL_PATH,
        prediction=None,
        error=None,
    )

# Form submission endpoint for demo page
@application.post("/predict-form")
def predict_form():
    message = (request.form.get("message") or "").strip()
    if not message:
        return render_template_string(
            DEMO_HTML,
            model_loaded=bool(_loaded_model is not None and _vectorizer is not None),
            model_path=MODEL_PATH,
            prediction=None,
            error="Field 'message' is required and must be non-empty.",
        ), 400
    try:
        label = _predict_text(message)
        return render_template_string(
            DEMO_HTML,
            model_loaded=True,
            model_path=MODEL_PATH,
            prediction=label,
            error=None,
        )
    except FileNotFoundError:
        return render_template_string(
            DEMO_HTML,
            model_loaded=False,
            model_path=MODEL_PATH,
            prediction=None,
            error="Model artifacts not found on server.",
        ), 503
    except Exception as e:
        logger.exception("Inference error: %s", e)
        return render_template_string(
            DEMO_HTML,
            model_loaded=bool(_loaded_model is not None and _vectorizer is not None),
            model_path=MODEL_PATH,
            prediction=None,
            error="Inference failed.",
        ), 500

# JSON API endpoint for predictions
@application.post("/predict")
def predict_json():
    data = request.get_json(silent=True) or {}
    message = str(data.get("message", "")).strip()
    if not message:
        return jsonify({"error": "Field 'message' is required and must be non-empty."}), 400
    try:
        label = _predict_text(message)
        return jsonify({"label": label}), 200
    except FileNotFoundError:
        return jsonify({"error": "Model artifacts not found on server."}), 503
    except Exception as e:
        logger.exception("Inference error: %s", e)
        return jsonify({"error": "Inference failed."}), 500


if __name__ == "__main__":
    # Local dev run; in EB, Gunicorn (from Procfile) will host the app
    port = int(os.getenv("PORT", "8000"))
    application.run(host="0.0.0.0", port=port, debug=False)
