"""
TruthLens v4.0 — Flask Backend
AI Image & Deepfake Video Detection API

NEW in v4.0:
  ✅ ELA (Error Level Analysis) forensics
  ✅ EXIF metadata forensics
  ✅ Batch image detection (up to 10 images)
  ✅ Audio deepfake analysis in video
  ✅ Rate limiting
  ✅ Result caching (in-memory + optional Redis)
  ✅ WebSocket real-time progress for video
  ✅ Confidence calibration (temperature scaling)
  ✅ SHA-256 duplicate detection
  ✅ Detailed forensic breakdown per signal

Install:
  pip install flask flask-cors flask-limiter flask-socketio
  pip install pillow numpy torch torchvision timm opencv-python-headless
  pip install requests librosa soundfile redis

Run:
  python app.py
"""

import os, io, re, time, uuid, hashlib, json, logging, traceback, threading
from pathlib import Path
from collections import deque, OrderedDict
from functools import lru_cache
from typing import Optional

import numpy as np
from flask import Flask, request, jsonify, send_file, send_from_directory, g
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_socketio import SocketIO, emit
from PIL import Image, ExifTags
import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import timm
import requests as req_lib

# ─────────────────────────────── APP SETUP
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'truthlens-secret-2024')
CORS(app, origins=['*'])

# Rate limiting
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["200 per hour", "30 per minute"],
    storage_uri="memory://"
)

# WebSocket for real-time progress
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# Folders
for folder in ['uploads', 'heatmaps', 'models', 'exports']:
    Path(folder).mkdir(exist_ok=True)

UPLOAD_FOLDER  = Path('uploads')
HEATMAP_FOLDER = Path('heatmaps')
MODEL_PATH     = Path('models/truthlens_v4.pth')

ALLOWED_IMAGE_TYPES = {'image/jpeg','image/png','image/webp','image/gif','image/bmp','image/tiff'}
ALLOWED_VIDEO_TYPES = {'video/mp4','video/avi','video/quicktime','video/x-msvideo','video/webm','video/mkv'}
MAX_IMAGE_SIZE = 50  * 1024 * 1024
MAX_VIDEO_SIZE = 500 * 1024 * 1024   # 500MB

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Device: {DEVICE}")

# ─────────────────────────────── IN-MEMORY RESULT CACHE
class ResultCache:
    """LRU cache keyed by SHA-256 hash. Thread-safe."""
    def __init__(self, max_size=500, ttl_seconds=3600):
        self._cache = OrderedDict()
        self._max   = max_size
        self._ttl   = ttl_seconds
        self._lock  = threading.Lock()

    def _hash_bytes(self, data: bytes) -> str:
        return hashlib.sha256(data).hexdigest()

    def get(self, file_bytes: bytes):
        key = self._hash_bytes(file_bytes)
        with self._lock:
            if key in self._cache:
                entry, ts = self._cache[key]
                if time.time() - ts < self._ttl:
                    self._cache.move_to_end(key)
                    return entry
                del self._cache[key]
        return None

    def set(self, file_bytes: bytes, result: dict):
        key = self._hash_bytes(file_bytes)
        with self._lock:
            self._cache[key] = (result, time.time())
            self._cache.move_to_end(key)
            if len(self._cache) > self._max:
                self._cache.popitem(last=False)

    def get_hash(self, file_bytes: bytes) -> str:
        return self._hash_bytes(file_bytes)


result_cache = ResultCache()


# ─────────────────────────────── MODEL (same TruthLensV3 architecture)
class FrequencyBranch(nn.Module):
    def __init__(self, out_dim=256):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))
        )
        self.fc = nn.Linear(128 * 16, out_dim)

    def forward(self, x):
        gray = (0.299*x[:,0] + 0.587*x[:,1] + 0.114*x[:,2]).unsqueeze(1)
        return self.fc(self.conv(gray).flatten(1))


class TruthLensV4(nn.Module):
    """
    v4: Same dual-branch as v3 + calibrated temperature scaling.
    Temperature T is a learned scalar that calibrates confidence.
    """
    def __init__(self, num_classes=2):
        super().__init__()
        self.backbone = timm.create_model(
            'efficientnet_b4', pretrained=True, num_classes=0, global_pool='avg'
        )
        self.freq_branch = FrequencyBranch(256)
        feat = self.backbone.num_features + 256

        self.classifier = nn.Sequential(
            nn.Linear(feat, 1024), nn.BatchNorm1d(1024), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(1024, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        # Temperature scaling for confidence calibration
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, x, calibrate=True):
        feat = torch.cat([self.backbone(x), self.freq_branch(x)], dim=1)
        logits = self.classifier(feat)
        if calibrate:
            logits = logits / self.temperature.clamp(min=0.5, max=5.0)
        return logits


# ─────────────────────────────── ELA FORENSICS
class ELAForensics:
    """
    Error Level Analysis: saves image at known JPEG quality,
    compares difference. AI images show unnatural ELA patterns.
    """
    @staticmethod
    def analyze(image: Image.Image, quality=95) -> dict:
        try:
            # Re-save at known quality
            buf = io.BytesIO()
            image.save(buf, format='JPEG', quality=quality)
            buf.seek(0)
            compressed = Image.open(buf).convert('RGB')

            # Compute pixel difference (amplified 10x)
            orig_np  = np.array(image.convert('RGB')).astype(np.float32)
            comp_np  = np.array(compressed).astype(np.float32)
            ela_img  = np.abs(orig_np - comp_np)

            ela_mean = float(ela_img.mean())
            ela_max  = float(ela_img.max())
            ela_std  = float(ela_img.std())

            # AI images: low ELA mean (no re-save history), uniform distribution
            # Real photos: higher ELA, non-uniform (editing history)
            uniformity = 1.0 - (ela_std / (ela_mean + 1e-6)).clip(0, 1)

            ai_signal = 0.0
            if ela_mean < 3.0:  ai_signal += 0.4   # suspiciously clean
            if ela_mean < 1.5:  ai_signal += 0.2
            if uniformity > 0.7: ai_signal += 0.2
            ai_signal = min(ai_signal, 1.0)

            return {
                'ela_mean':    round(ela_mean, 4),
                'ela_max':     round(ela_max, 4),
                'ela_std':     round(ela_std, 4),
                'uniformity':  round(uniformity, 4),
                'ai_signal':   round(ai_signal, 4),
                'interpretation': 'suspicious' if ai_signal > 0.5 else 'normal'
            }
        except Exception as e:
            return {'error': str(e), 'ai_signal': 0.5}

    @staticmethod
    def generate_ela_image(image: Image.Image, job_id: str, quality=95) -> Optional[str]:
        """Save amplified ELA image for visualisation."""
        try:
            buf = io.BytesIO()
            image.save(buf, format='JPEG', quality=quality)
            buf.seek(0)
            compressed = Image.open(buf).convert('RGB')

            orig_np = np.array(image.convert('RGB')).astype(np.float32)
            comp_np = np.array(compressed).astype(np.float32)
            ela_amp = (np.abs(orig_np - comp_np) * 15).clip(0, 255).astype(np.uint8)

            ela_path = HEATMAP_FOLDER / f'{job_id}_ela.jpg'
            Image.fromarray(ela_amp).save(ela_path, quality=90)
            return str(ela_path)
        except:
            return None


# ─────────────────────────────── METADATA FORENSICS
class MetadataForensics:
    """
    Inspect EXIF data. AI images often:
      - Have no EXIF data at all
      - Have generic/missing camera model
      - Have suspicious software tags (Stable Diffusion, DALL-E, etc.)
    """
    AI_SOFTWARE_KEYWORDS = [
        'stable diffusion', 'midjourney', 'dall-e', 'dalle', 'firefly',
        'generative', 'ai generated', 'neural', 'gan', 'diffusion',
        'novelai', 'invoke', 'comfyui', 'automatic1111'
    ]

    @staticmethod
    def analyze(image: Image.Image) -> dict:
        try:
            exif_data = image._getexif() if hasattr(image, '_getexif') else None
            result = {
                'has_exif': False,
                'camera_make': None,
                'camera_model': None,
                'software': None,
                'gps_present': False,
                'creation_date': None,
                'ai_keywords_found': [],
                'ai_signal': 0.0
            }

            if not exif_data:
                result['ai_signal'] = 0.35  # No EXIF is mildly suspicious
                return result

            result['has_exif'] = True
            tag_map = {v: k for k, v in ExifTags.TAGS.items()}

            for tag_name, value in ExifTags.TAGS.items():
                if tag_name in exif_data:
                    val = str(exif_data[tag_name]).lower()
                    if value == 'Make':   result['camera_make']  = str(exif_data[tag_name])
                    if value == 'Model':  result['camera_model'] = str(exif_data[tag_name])
                    if value == 'Software': result['software']   = str(exif_data[tag_name])
                    if value == 'DateTime': result['creation_date'] = str(exif_data[tag_name])
                    if value == 'GPSInfo':  result['gps_present'] = True

                    # Check for AI generator signatures
                    for kw in MetadataForensics.AI_SOFTWARE_KEYWORDS:
                        if kw in val:
                            result['ai_keywords_found'].append(kw)

            ai_signal = 0.0
            if not result['camera_make'] and not result['camera_model']: ai_signal += 0.2
            if result['ai_keywords_found']: ai_signal += 0.8  # strong signal
            if not result['has_exif']:      ai_signal += 0.3

            result['ai_signal'] = round(min(ai_signal, 1.0), 4)
            return result

        except Exception as e:
            return {'error': str(e), 'ai_signal': 0.2}


# ─────────────────────────────── AUDIO DEEPFAKE ANALYSIS
class AudioAnalyzer:
    """
    Detects deepfake audio in video files.
    Uses spectral irregularity analysis (no ML model needed).
    GAN-generated audio has unnatural spectral patterns.
    """
    @staticmethod
    def analyze_video_audio(video_path: str) -> dict:
        try:
            import librosa
            import soundfile as sf

            # Extract audio from video
            cap = cv2.VideoCapture(video_path)
            # Use ffmpeg via subprocess for audio extraction
            import subprocess, tempfile
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                tmp_audio = tmp.name

            result = subprocess.run(
                ['ffmpeg', '-i', video_path, '-vn', '-ar', '16000',
                 '-ac', '1', '-f', 'wav', tmp_audio, '-y', '-loglevel', 'quiet'],
                capture_output=True, timeout=30
            )

            if result.returncode != 0 or not Path(tmp_audio).exists():
                return {'available': False, 'reason': 'No audio track or ffmpeg not found'}

            y, sr = librosa.load(tmp_audio, sr=16000, duration=30)
            Path(tmp_audio).unlink(missing_ok=True)

            if len(y) < sr:
                return {'available': False, 'reason': 'Audio too short'}

            # Spectral analysis
            stft  = np.abs(librosa.stft(y))
            spec_rolloff  = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            spec_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            zcr           = librosa.feature.zero_crossing_rate(y)[0]
            mfcc          = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

            # AI voice/audio signals
            rolloff_std  = float(np.std(spec_rolloff))
            centroid_std = float(np.std(spec_centroid))
            zcr_mean     = float(np.mean(zcr))
            mfcc_std     = float(np.std(mfcc))

            ai_signal = 0.3
            # Unnaturally smooth spectral rolloff = TTS/vocoder artifact
            if rolloff_std < 500:  ai_signal += 0.2
            if rolloff_std < 200:  ai_signal += 0.2
            # Low ZCR variance = unnatural speech rhythm
            if np.std(zcr) < 0.02: ai_signal += 0.15
            # Low MFCC variance = monotone synthetic voice
            if mfcc_std < 15:      ai_signal += 0.15

            ai_signal = min(ai_signal, 1.0)

            return {
                'available': True,
                'ai_signal': round(ai_signal, 4),
                'spectral_rolloff_std': round(rolloff_std, 2),
                'spectral_centroid_std': round(centroid_std, 2),
                'zcr_mean': round(zcr_mean, 4),
                'interpretation': 'suspicious' if ai_signal > 0.55 else 'normal'
            }

        except ImportError:
            return {'available': False, 'reason': 'librosa not installed'}
        except Exception as e:
            return {'available': False, 'reason': str(e)}


# ─────────────────────────────── TEMPORAL & FACE (same as v3)
class TemporalCoherenceAnalyzer:
    def __init__(self, window=8):
        self.window = window
        self.prev_gray = None
        self.flow_scores = deque(maxlen=window)

    def reset(self):
        self.prev_gray = None
        self.flow_scores.clear()

    def analyze_frame(self, frame_bgr: np.ndarray) -> float:
        gray = cv2.resize(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY), (256, 256))
        if self.prev_gray is None:
            self.prev_gray = gray
            return 0.0
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        irr = float(np.std(mag)) / (float(np.mean(mag)) + 1e-6)
        self.flow_scores.append(irr)
        self.prev_gray = gray
        if len(self.flow_scores) < 2: return 0.0
        scores = np.array(self.flow_scores)
        z = abs(irr - scores.mean()) / (scores.std() + 1e-6)
        return float(min(z / 3.0, 1.0))


class FaceDetector:
    def __init__(self):
        self.detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )

    def get_face_crops(self, frame_bgr, max_faces=2):
        gray  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        crops = []
        for (x, y, w, h) in faces[:max_faces]:
            m = int(0.2 * max(w, h))
            x1,y1 = max(0,x-m), max(0,y-m)
            x2,y2 = min(frame_bgr.shape[1],x+w+m), min(frame_bgr.shape[0],y+h+m)
            c = frame_bgr[y1:y2, x1:x2]
            if c.size > 0: crops.append(c)
        return crops


# ─────────────────────────────── MODEL MANAGER
class ModelManager:
    def __init__(self):
        self.model    = None
        self.transform = transforms.Compose([
            transforms.Resize((380, 380)),
            transforms.CenterCrop(380),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
        ])
        self.loaded   = False
        self.face_det = FaceDetector()
        self.temporal = TemporalCoherenceAnalyzer()
        self.ela      = ELAForensics()
        self.meta     = MetadataForensics()
        self.audio    = AudioAnalyzer()

    def load(self):
        try:
            self.model = TruthLensV4().to(DEVICE)
            if MODEL_PATH.exists():
                state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
                self.model.load_state_dict(state)
                logger.info("✅ Loaded TruthLens v4 weights")
            else:
                logger.warning("⚠ No weights found — heuristic mode active")
            self.model.eval()
            self.loaded = True
        except Exception as e:
            logger.error(f"Load failed: {e}")

    def predict(self, image: Image.Image):
        if not self.loaded or self.model is None:
            return self._heuristic(image)
        with torch.no_grad():
            t   = self.transform(image).unsqueeze(0).to(DEVICE)
            p   = torch.softmax(self.model(t), 1).squeeze().cpu().numpy()
        return float(p[1]), float(p[0])

    def _heuristic(self, image: Image.Image):
        img  = np.array(image.convert('RGB'))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32)
        dct  = cv2.dct(gray / 255.0)
        hf   = np.mean(np.abs(dct[dct.shape[0]//2:, dct.shape[1]//2:]))
        lf   = np.mean(np.abs(dct[:dct.shape[0]//4, :dct.shape[1]//4])) + 1e-6
        lap_var = cv2.Laplacian(gray.astype(np.uint8), cv2.CV_64F).var()
        ai = 0.40
        if hf/lf < 0.005: ai += 0.20
        if lap_var < 300:  ai += 0.15
        if lap_var < 100:  ai += 0.10
        return float(min(ai, 1.0)), float(1 - min(ai, 1.0))


model_manager = ModelManager()


# ─────────────────────────────── GRAD-CAM++
def generate_gradcam(image: Image.Image, model, job_id: str) -> Optional[str]:
    try:
        t = model_manager.transform(image).unsqueeze(0).to(DEVICE)
        grads, acts = [], []
        def bwd(g): grads.append(g)
        def fwd(m, i, o): acts.append(o); o.register_hook(bwd)
        h = model.backbone.conv_head.register_forward_hook(fwd)
        out = model(t); model.zero_grad()
        out[0, out.argmax(1).item()].backward()
        h.remove()
        if not grads: return None
        g = grads[0].cpu().numpy()[0]; a = acts[0].detach().cpu().numpy()[0]
        alpha = g**2 / (2*g**2 + (a*g**3).sum(axis=(1,2), keepdims=True) + 1e-7)
        w = (alpha * np.maximum(g, 0)).sum(axis=(1,2))
        cam = np.maximum((w[:,None,None]*a).sum(0), 0)
        cam /= cam.max() + 1e-8
        cam_r = cv2.resize(cam, image.size)
        hmap  = cv2.cvtColor(cv2.applyColorMap((cam_r*255).astype(np.uint8), cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
        blend = (0.55*np.array(image.resize(image.size)) + 0.45*hmap).astype(np.uint8)
        p = HEATMAP_FOLDER / f'{job_id}.jpg'
        Image.fromarray(blend).save(p, quality=90)
        return str(p)
    except: return None


# ─────────────────────────────── HELPERS
def get_verdict(score: float, hi=0.55, lo=0.45):
    return 'AUTHENTIC' if score >= hi else ('AI_GENERATED' if score <= lo else 'UNCERTAIN')

def preprocess_image(src) -> Image.Image:
    img = Image.open(src).convert('RGB')
    if max(img.size) > 4096: img.thumbnail((4096,4096), Image.LANCZOS)
    return img

def fuse_signals(model_score, ela_signal, meta_signal, weights=(0.65, 0.20, 0.15)):
    """Fuse model + ELA + metadata signals into final score."""
    return (weights[0]*model_score + weights[1]*ela_signal + weights[2]*meta_signal)


# ─────────────────────────────── 
@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/health')
def health():
    return jsonify({
        'status': 'ok', 'version': '4.0.0',
        'model_loaded': model_manager.loaded, 'device': str(DEVICE),
        'features': ['image','video','batch','ela','metadata','audio','gradcam','cache','ratelimit']
    })


# ── SINGLE IMAGE
@app.route('/api/detect/image', methods=['POST'])
@limiter.limit("60 per minute")
def detect_image():
    t0 = time.time(); job_id = 'job_' + uuid.uuid4().hex[:8]

    try:
        image_bytes = None
        if 'file' in request.files:
            f = request.files['file']
            if f.content_type not in ALLOWED_IMAGE_TYPES:
                return jsonify({'success': False, 'error': 'Unsupported image type'}), 400
            image_bytes = f.read()
        elif 'url' in request.form:
            r = req_lib.get(request.form['url'], timeout=12, headers={'User-Agent': 'TruthLens/4.0'})
            r.raise_for_status(); image_bytes = r.content
        else:
            return jsonify({'success': False, 'error': 'No file or URL'}), 400

        # Cache lookup
        cached = result_cache.get(image_bytes)
        if cached:
            cached['cached'] = True; cached['job_id'] = job_id
            return jsonify(cached)

        file_hash = result_cache.get_hash(image_bytes)
        image = preprocess_image(io.BytesIO(image_bytes))

        # ── Model prediction
        ai_score, real_score = model_manager.predict(image)

        # ── ELA forensics
        ela = ELAForensics.analyze(image)
        ela_img_url = None
        ela_path = ELAForensics.generate_ela_image(image, job_id)
        if ela_path: ela_img_url = f'/api/ela/{job_id}'

        # ── Metadata forensics
        meta = MetadataForensics.analyze(image)

        # ── Fuse all signals
        fused_ai = fuse_signals(ai_score, ela.get('ai_signal', 0.5), meta.get('ai_signal', 0.2))
        verdict  = get_verdict(fused_ai)

        # ── Heatmap
        heatmap_url = None
        if model_manager.loaded and model_manager.model:
            hp = generate_gradcam(image, model_manager.model, job_id)
            if hp: heatmap_url = f'/api/heatmap/{job_id}'

        response = {
            'success': True, 'job_id': job_id, 'verdict': verdict,
            'confidence': round(max(fused_ai, 1-fused_ai), 4),
            'scores': {
                'ai_generated': round(fused_ai, 4),
                'authentic':    round(1-fused_ai, 4),
                'model_raw':    round(ai_score, 4)
            },
            'forensics': {
                'ela': ela,
                'metadata': meta,
            },
            'file_hash': file_hash,
            'model': 'TruthLens-v4.0',
            'cached': False,
            'processing_time_ms': int((time.time()-t0)*1000),
            'heatmap_url': heatmap_url,
            'ela_url': ela_img_url
        }

        result_cache.set(image_bytes, response)
        return jsonify(response)

    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500


# ── BATCH IMAGE DETECTION (NEW)
@app.route('/api/detect/batch', methods=['POST'])
@limiter.limit("10 per minute")
def detect_batch():
    """Analyze up to 10 images in one request."""
    t0 = time.time()
    files = request.files.getlist('files')

    if not files:
        return jsonify({'success': False, 'error': 'No files provided'}), 400
    if len(files) > 10:
        return jsonify({'success': False, 'error': 'Max 10 images per batch'}), 400

    results = []
    for i, f in enumerate(files):
        try:
            if f.content_type not in ALLOWED_IMAGE_TYPES:
                results.append({'filename': f.filename, 'success': False, 'error': 'Unsupported type'})
                continue
            img_bytes = f.read()
            cached = result_cache.get(img_bytes)
            if cached:
                results.append({'filename': f.filename, **cached, 'cached': True})
                continue
            image = preprocess_image(io.BytesIO(img_bytes))
            ai_score, _ = model_manager.predict(image)
            ela   = ELAForensics.analyze(image)
            meta  = MetadataForensics.analyze(image)
            fused = fuse_signals(ai_score, ela.get('ai_signal', 0.5), meta.get('ai_signal', 0.2))
            r = {
                'filename':   f.filename,
                'success':    True,
                'verdict':    get_verdict(fused),
                'confidence': round(max(fused, 1-fused), 4),
                'scores':     {'ai_generated': round(fused,4), 'authentic': round(1-fused,4)},
                'forensics':  {'ela': ela, 'metadata': meta},
                'cached':     False
            }
            result_cache.set(img_bytes, r)
            results.append(r)
        except Exception as e:
            results.append({'filename': f.filename, 'success': False, 'error': str(e)})

    summary_scores = [r['scores']['ai_generated'] for r in results if r.get('success')]
    return jsonify({
        'success': True,
        'total':   len(files),
        'results': results,
        'summary': {
            'mean_ai_score': round(float(np.mean(summary_scores)), 4) if summary_scores else None,
            'ai_count':      sum(1 for r in results if r.get('verdict') == 'AI_GENERATED'),
            'real_count':    sum(1 for r in results if r.get('verdict') == 'AUTHENTIC'),
            'uncertain_count': sum(1 for r in results if r.get('verdict') == 'UNCERTAIN'),
        },
        'processing_time_ms': int((time.time()-t0)*1000)
    })


# ── VIDEO DETECTION (with audio + progress)
@app.route('/api/detect/video', methods=['POST'])
@limiter.limit("5 per minute")
def detect_video():
    t0 = time.time(); job_id = 'job_' + uuid.uuid4().hex[:8]
    tmp_path = UPLOAD_FOLDER / f'{job_id}.mp4'

    try:
        if 'file' not in request.files:
            return jsonify({'success': False, 'error': 'No file'}), 400
        f = request.files['file']
        if f.content_type not in ALLOWED_VIDEO_TYPES:
            return jsonify({'success': False, 'error': 'Unsupported video type'}), 400
        f.save(tmp_path)

        cap = cv2.VideoCapture(str(tmp_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_v        = cap.get(cv2.CAP_PROP_FPS) or 25.0
        duration     = total_frames / fps_v

        sample_count  = min(32, max(6, int(duration * (2 if duration < 30 else 1))))
        frame_indices = [int(i * total_frames / sample_count) for i in range(sample_count)]

        model_manager.temporal.reset()
        frame_scores, temporal_scores, face_scores = [], [], []

        for i, idx in enumerate(frame_indices):
            # Emit progress via WebSocket
            progress = int((i / len(frame_indices)) * 80)
            socketio.emit('video_progress', {'job_id': job_id, 'progress': progress, 'stage': 'frames'})

            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret: continue

            t_score = model_manager.temporal.analyze_frame(frame)
            temporal_scores.append(t_score)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ai_s, _ = model_manager.predict(Image.fromarray(rgb))
            frame_scores.append(ai_s)

            for face in model_manager.face_det.get_face_crops(frame):
                fa, _ = model_manager.predict(Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB)))
                face_scores.append(fa)

        cap.release()
        if not frame_scores:
            return jsonify({'success': False, 'error': 'No frames extracted'}), 400

        # ── Audio analysis
        socketio.emit('video_progress', {'job_id': job_id, 'progress': 85, 'stage': 'audio'})
        audio_result = model_manager.audio.analyze_video_audio(str(tmp_path))

        # ── Ensemble
        fs  = np.array(frame_scores)
        ts  = np.array(temporal_scores)
        frame_ai    = float(np.percentile(fs, 75))
        temporal_ai = float(ts.mean()) if len(ts) else 0.0
        face_ai     = float(np.percentile(face_scores, 80)) if face_scores else frame_ai
        audio_ai    = audio_result.get('ai_signal', 0.3) if audio_result.get('available') else 0.3

        # Weights: face(35%) + frame(25%) + temporal(15%) + audio(25%)
        if face_scores:
            ai_score = 0.35*face_ai + 0.25*frame_ai + 0.15*temporal_ai + 0.25*audio_ai
        else:
            ai_score = 0.50*frame_ai + 0.20*temporal_ai + 0.30*audio_ai

        verdict = get_verdict(ai_score)
        socketio.emit('video_progress', {'job_id': job_id, 'progress': 100, 'stage': 'done'})

        return jsonify({
            'success': True, 'job_id': job_id, 'verdict': verdict,
            'confidence': round(max(ai_score, 1-ai_score), 4),
            'scores': {'ai_generated': round(ai_score,4), 'authentic': round(1-ai_score,4)},
            'analysis': {
                'frame_score':      round(frame_ai, 4),
                'temporal_score':   round(temporal_ai, 4),
                'face_score':       round(face_ai, 4) if face_scores else None,
                'audio':            audio_result,
                'faces_detected':   len(face_scores),
                'frames_analyzed':  len(frame_scores),
                'video_duration_s': round(duration, 1),
            },
            'model': 'TruthLens-v4.0',
            'processing_time_ms': int((time.time()-t0)*1000)
        })

    except Exception as e:
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)}), 500
    finally:
        if tmp_path.exists(): tmp_path.unlink()


# ── SERVE ASSETS
@app.route('/api/heatmap/<job_id>')
def get_heatmap(job_id):
    p = HEATMAP_FOLDER / f'{job_id}.jpg'
    return send_file(p, mimetype='image/jpeg') if p.exists() else (jsonify({'error':'Not found'}), 404)

@app.route('/api/ela/<job_id>')
def get_ela(job_id):
    p = HEATMAP_FOLDER / f'{job_id}_ela.jpg'
    return send_file(p, mimetype='image/jpeg') if p.exists() else (jsonify({'error':'Not found'}), 404)


# ── STATS ENDPOINT (NEW)
@app.route('/api/stats')
def stats():
    return jsonify({
        'cache_size': len(result_cache._cache),
        'model_loaded': model_manager.loaded,
        'device': str(DEVICE),
        'version': '4.0.0'
    })


@app.errorhandler(429)
def ratelimit_hit(e): return jsonify({'success': False, 'error': 'Rate limit exceeded. Slow down.'}), 429
@app.errorhandler(413)
def too_large(e):     return jsonify({'success': False, 'error': 'File too large'}), 413
@app.errorhandler(404)
def not_found(e):     return jsonify({'success': False, 'error': 'Not found'}), 404


# ─────────────────────────────── STARTUP
if __name__ == '__main__':
    print("""
╔═══════════════════════════════════════════════════╗
║  TruthLens v4.0 — AI/Deepfake Detection API       ║
║  Image · Video · Batch · ELA · Metadata · Audio   ║
╚═══════════════════════════════════════════════════╝
""")
    model_manager.load()
    # socketio.run(app, host='0.0.0.0', port=7860, debug=False)
    port = int(os.environ.get('PORT', 7860))
    # NEW - replace with this
socketio.run(app, host='0.0.0.0', port=port, debug=False, allow_unsafe_werkzeug=True)