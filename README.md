# TruthLens v4.0 — AI & Deepfake Detection

Final Year Project — AI Image and Video Detection Platform

## Quick Start

### 1. Install packages
```cmd
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Check setup
```cmd
python check_setup.py
```

### 3. Run app
```cmd
python app.py
```

### 4. Open dashboard
```
http://localhost:5000/static/dashboard.html
```

---

## All Commands

| Command | What it does |
|---------|-------------|
| `python check_setup.py` | Verify all packages and folders are ready |
| `python app.py` | Start the detection server |
| `python train.py --data ./data --epochs 20 --batch 8 --workers 0` | Train the model |
| `python evaluate_model.py --data ./data` | Test model accuracy |
| `python export_model.py` | Export to ONNX for faster inference |
| `python test_api.py` | Test all API endpoints |
| `python split_data.py --real path/to/real --fake path/to/fake` | Split your images into train/val |

---

## Project Structure
```
TruthLens/
├── app.py                 — Main Flask backend
├── train.py               — Model training script
├── evaluate_model.py      — Evaluate model accuracy
├── export_model.py        — Export to ONNX
├── test_api.py            — API endpoint tests
├── check_setup.py         — Verify installation
├── split_data.py          — Organise dataset
├── requirements.txt       — Python packages
├── README.md              — This file
├── .env                   — Configuration
├── static/
│   └── dashboard.html     — Web UI
├── models/
│   └── truthlens_v4.pth   — Trained weights
├── data/
│   ├── train/real/        — Real training images
│   ├── train/fake/        — AI training images
│   ├── val/real/          — Real validation images
│   └── val/fake/          — AI validation images
├── uploads/               — Temp video storage
├── heatmaps/              — Grad-CAM outputs
└── exports/               — ONNX models + reports
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Server status |
| POST | `/api/detect/image` | Detect AI in image |
| POST | `/api/detect/batch` | Detect AI in up to 10 images |
| POST | `/api/detect/video` | Detect deepfake in video |
| GET | `/api/heatmap/<id>` | Get Grad-CAM heatmap |
| GET | `/api/ela/<id>` | Get ELA forensic image |
| GET | `/api/stats` | Server statistics |

---

## Training Data

Download CIFAKE dataset:
https://www.kaggle.com/datasets/birdy654/cifake-real-and-ai-generated-synthetic-images

---

## Tech Stack

- Flask — Web server
- PyTorch + EfficientNet-B4 — AI model
- DCT Frequency Analysis — Signal detection
- Grad-CAM++ — Visual explainability
- ELA Forensics — Error level analysis
- OpenCV — Video processing
- React — Dashboard UI
```

Press **`Ctrl + S`**

---

## ✅ NOW Complete Folder Structure
```
📁 TruthLens
 ├── 📄 app.py                ✅
 ├── 📄 train.py              ✅
 ├── 📄 check_setup.py        ✅ NEW
 ├── 📄 evaluate_model.py     ✅ NEW
 ├── 📄 export_model.py       ✅ NEW
 ├── 📄 test_api.py           ✅ NEW
 ├── 📄 split_data.py         ✅ NEW
 ├── 📄 requirements.txt      ✅ NEW
 ├── 📄 README.md             ✅ NEW
 ├── 📄 .env                  ✅ NEW
 ├── 📁 static
 │    └── 📄 dashboard.html   ✅
 ├── 📁 models                ✅
 ├── 📁 data
 │    ├── 📁 train
 │    │    ├── 📁 real        ✅
 │    │    └── 📁 fake        ✅
 │    └── 📁 val
 │         ├── 📁 real        ✅
 │         └── 📁 fake        ✅
 ├── 📁 uploads               ✅
 ├── 📁 heatmaps              ✅
 ├── 📁 exports               ✅
 └── 📁 venv                  ✅


 to run this first to checck everything

 python check_setup.py