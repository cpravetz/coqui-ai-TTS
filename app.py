from flask import Flask, request, send_file, abort, render_template, render_template_string
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer
from urllib.parse import parse_qs
from threading import Lock
import logging
import torch
import json
import io
import os

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Set up the correct template directory
template_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
app = Flask(__name__, template_folder=template_dir)

# Get the secret token from environment variable
SECRET_TOKEN = os.environ.get('ACCESS_TOKEN')
if not SECRET_TOKEN:
    raise ValueError("ACCESS_TOKEN environment variable must be set")

# Initialize model manager and synthesizer
model_manager = ModelManager()
model_path, config_path, model_item = model_manager.download_model("tts_models/en/ljspeech/vits")

# Synthesizer initialization parameters
synthesizer_params = {
    'tts_checkpoint': model_path,
    'tts_config_path': config_path,
    'tts_speakers_file': None,
    'tts_languages_file': None,
    'vocoder_checkpoint': None,
    'vocoder_config': None,
    'encoder_checkpoint': None,
    'encoder_config': None,
    'use_cuda': torch.cuda.is_available()
}

try:
    synthesizer = Synthesizer(**synthesizer_params)
except Exception as e:
    logger.error(f"Could not initialize synthesizer: {e}")
    synthesizer = None

# Create a Lock instance for thread safety
lock = Lock()

def verify_token():
    """Verify the authorization token"""
    auth_header = request.headers.get('Authorization')
    if not auth_header:
        return False
    try:
        token_type, token = auth_header.split()
        if token_type.lower() != 'bearer':
            return False
        return token == SECRET_TOKEN
    except ValueError:
        return False

@app.before_request
def check_auth():
    """Check authentication before each request"""
    # Skip auth for index and details pages
    if request.endpoint in [ 'index', 'health_check', 'health', 'details' ]:
        return
    if not verify_token():
        abort(403, description="Invalid or missing authentication token")

def style_wav_uri_to_dict(style_wav: str) -> str | dict:
    """Transform an uri style_wav, in either a string or a dict"""
    if style_wav:
        if os.path.isfile(style_wav) and style_wav.endswith(".wav"):
            return style_wav
        return json.loads(style_wav)
    return None

@app.route("/")
def health_check():
    return "OK", 200

@app.route("/health")
def health():
    """Dedicated health check endpoint that doesn't interact with TTS"""
    return {"status": "healthy"}, 200

@app.route("/details")
def details():
    if not synthesizer:
        return {"error": "Synthesizer not initialized"}, 500
    
    model_config = synthesizer.tts_config
    vocoder_config = synthesizer.vocoder_config or None

    return render_template(
        "details.html",
        show_details=True,
        model_config=model_config,
        vocoder_config=vocoder_config,
        args={"model_name": "tts_models/en/ljspeech/vits"}
    )

@app.route("/api/tts", methods=["GET", "POST"])
def tts():
    if not synthesizer:
        return {"error": "Synthesizer not initialized"}, 500

    with lock:
        # Handle JSON body for POST requests
        if request.method == "POST" and request.is_json:
            data = request.get_json()
            text = data.get("text", "")
            speaker_idx = data.get("speaker", "")
            language_idx = data.get("language", "")
            style_wav = data.get("styleWav", "")
        else:
            # Fallback to query parameters or form data
            text = request.values.get("text", "")
            speaker_idx = request.values.get("speaker_id", "")
            language_idx = request.values.get("language_id", "")
            style_wav = request.values.get("style_wav", "")

        if not text:
            return {"error": "No text provided"}, 400
            
        # Process style_wav if present
        if style_wav:
            style_wav = style_wav_uri_to_dict(style_wav)

        logger.info("Model input: %s", text)
        logger.info("Speaker idx: %s", speaker_idx)
        logger.info("Language idx: %s", language_idx)
        
        wavs = synthesizer.tts(text)
        out = io.BytesIO()
        synthesizer.save_wav(wavs, out)
        out.seek(0)
        
    return send_file(out, mimetype="audio/wav")

@app.route("/locales", methods=["GET"])
def mary_tts_api_locales():
    """MaryTTS-compatible /locales endpoint"""
    return render_template_string("{{ locale }}\n", locale="en")

@app.route("/voices", methods=["GET"])
def mary_tts_api_voices():
    """MaryTTS-compatible /voices endpoint"""
    return render_template_string(
        "{{ name }} {{ locale }} {{ gender }}\n", 
        name="vits", 
        locale="en", 
        gender="u"
    )

@app.route("/process", methods=["GET", "POST"])
def mary_tts_api_process():
    """MaryTTS-compatible /process endpoint"""
    if not synthesizer:
        return {"error": "Synthesizer not initialized"}, 500

    with lock:
        if request.method == "POST":
            data = parse_qs(request.get_data(as_text=True))
            text = data.get("INPUT_TEXT", [""])[0]
        else:
            text = request.args.get("INPUT_TEXT", "")
        
        if not text:
            return {"error": "No text provided"}, 400
            
        logger.info("Model input: %s", text)
        wavs = synthesizer.tts(text)
        out = io.BytesIO()
        synthesizer.save_wav(wavs, out)
        out.seek(0)
        
    return send_file(out, mimetype="audio/wav")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
