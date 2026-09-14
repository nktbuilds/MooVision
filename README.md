# MooVision: Cow Segmentation & Behaviour Analysis (YOLO26 + SigLIP2)

Welcome to the companion repository for our video: https://youtu.be/u_ndIfISOKE

This repository contains interactive Streamlit applications for livestock monitoring. The project combines YOLO26 instance segmentation for precise cow detection with Google’s SigLIP2 vision-language model for zero-shot posture and behavior classification (such as lying, standing, or eating).

---

## Repository Contents

* **`app_yolo26n_seg.py`**: Lightweight interactive dashboard using the fast **YOLO26 Nano** segmentation model for high-FPS cow detection and count tallying.
* **`app_yolo26l_seg.py`**: Balanced dashboard utilizing **YOLO26 Large** segmentation for higher precision masking in dense farm environments.
* **`app_yolo26x_seg.py`**: Heavyweight dashboard leveraging **YOLO26 Extra Large** segmentation for maximum mask accuracy on low-resolution or overlapping livestock feeds.
* **`app_yolo26x_seg_sigLIP.py`**: The complete pipeline combining **YOLO26x-Seg** detection with **SigLIP2 (`google/siglip2-base-patch16-256`)**. It crops individual cows, runs them through the image encoder, and calculates cosine similarity against pre-encoded text behavior prompts.

---

## Technical Overview

1. **Instance Segmentation:** Standard bounding boxes often capture irrelevant background noise (e.g., stalls, feeding troughs). YOLO26 segmentation isolates the precise pixel mask of each cow (COCO class 19).
2. **Dynamic Bounding Box Padding:** Bounding boxes are cropped with adjustable percentage padding to preserve essential behavioral context (such as head placement near feed fences).
3. **Pre-Computed Text Embeddings:** Behavior prompts ("lying in stall", "eating green grass", "standing in aisle") are encoded once into RAM at startup via SigLIP2.
4. **Zero-Shot Crop Inference:** Each detected cow crop only passes through SigLIP's image encoder. The resulting vector is multiplied against the cached text matrix to assign real-time behavior probabilities without task-specific retraining.

---

## Quick Start & Setup

Run these commands in your terminal to set up the environment and start the dashboard:

```bash
# Initialize git repository (optional)
git init

# Configure .gitignore
echo "venv/" >> .gitignore
echo "__pycache__/" >> .gitignore
echo "*.pt" >> .gitignore
echo ".env" >> .gitignore

# Set up virtual environment
python3 -m venv venv
source venv/bin/activate

# Step 1: Install CPU-only PyTorch (reduces download size)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Step 2: Install dependencies from PyPI
pip install streamlit opencv-python numpy pillow ultralytics transformers

# Step 3: Launch the application (change filename to test different models)
streamlit run app_yolo26x_seg_sigLIP.py
```

---

## Useful Links & Resources

* **Companion YouTube Video:** https://youtu.be/u_ndIfISOKE
* **Google SigLIP2 Models (Hugging Face):** https://huggingface.co/collections/google/siglip2
* **OpenAI CLIP Conceptual Overview:** https://openai.com/index/clip/
* **Ultralytics YOLO Documentation:** https://docs.ultralytics.com/
* **OpenCV Library:** https://opencv.org/