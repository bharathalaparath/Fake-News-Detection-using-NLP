import logging

from flask import Blueprint, current_app, render_template, request

from preprocessing import clean_text

log = logging.getLogger(__name__)

bp = Blueprint("main", __name__)

MAX_TEXT_LENGTH = 20_000


@bp.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    submitted_text = ""

    if request.method == "POST":
        submitted_text = request.form.get("txt", "").strip()
        model = current_app.config.get("MODEL")

        if not submitted_text:
            error = "Please enter some article text to check."
        elif len(submitted_text) > MAX_TEXT_LENGTH:
            error = f"Text is too long (max {MAX_TEXT_LENGTH} characters)."
        elif model is None:
            error = "Model is not available right now. Please try again later."
        else:
            try:
                cleaned = clean_text(submitted_text)
                result = int(model.predict([cleaned])[0])
            except Exception:
                log.exception("Prediction failed")
                error = "Something went wrong while analyzing the text. Please try again."

    return render_template(
        "index.html", result=result, error=error, submitted_text=submitted_text
    )


@bp.route("/health")
def health():
    model_loaded = current_app.config.get("MODEL") is not None
    status_code = 200 if model_loaded else 503
    return {"status": "ok" if model_loaded else "degraded", "model_loaded": model_loaded}, status_code
