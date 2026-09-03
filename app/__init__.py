"""Flask application factory for the fake news detector."""
import logging
import os
from pathlib import Path
from typing import Optional

import joblib
from flask import Flask

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def create_app(model_path: Optional[str] = None) -> Flask:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    app = Flask(__name__, template_folder=str(PROJECT_ROOT / "templates"))

    resolved_model_path = Path(model_path or os.environ.get("MODEL_PATH", "model.pkl"))
    try:
        app.config["MODEL"] = joblib.load(resolved_model_path)
        log.info("Loaded model pipeline from %s", resolved_model_path)
    except FileNotFoundError:
        app.config["MODEL"] = None
        log.warning(
            "No model found at %s. Train one with training/train.py (after preparing data "
            "with scripts/prepare_data.py) or set MODEL_PATH to point at an existing pipeline. "
            "The app will run but predictions will be unavailable until then.",
            resolved_model_path,
        )

    from .routes import bp

    app.register_blueprint(bp)

    return app
