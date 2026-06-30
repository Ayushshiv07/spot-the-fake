"""
app.py — Flask live demo server.

Features a highly unique, interactive 3D Computer Vision Dashboard:
  - Real-time interactive 3D FFT terrain mesh visualization using Three.js.
  - Floating 3D particle background responsive to mouse movement.
  - 3D card tilt effects (vanilla-tilt.js).
  - Laplacian edge-detection maps.
  - Detailed explainable diagnostics of all 13 features.
  - Tested history list to compare previous predictions.

Usage:
    python app.py
    Then open http://localhost:5000 in your browser.
"""

import os
import tempfile
import base64
import time
import io
import numpy as np
from PIL import Image
from scipy.ndimage import laplace
from flask import Flask, request, jsonify, render_template_string

from predict import predict
from features import extract_features, FEATURE_NAMES

app = Flask("FakePhotoDetector")

# ── helper function for 3D FFT downsampled grid ────────────────────────────────
def get_fft_3d_grid(image_path: str, grid_size: int = 32) -> list:
    """Generate a downsampled 32x32 grid of FFT magnitude values for 3D mesh rendering."""
    try:
        img = Image.open(image_path).convert("L")
        w, h = img.size
        size = 256
        if w <= size or h <= size:
            crop_img = img.resize((size, size))
        else:
            x0 = (w - size) // 2
            y0 = (h - size) // 2
            crop_img = img.crop((x0, y0, x0 + size, y0 + size))
        arr = np.array(crop_img, dtype=np.float32)
        fft = np.fft.fft2(arr)
        fft_shift = np.fft.fftshift(fft)
        mag = np.log1p(np.abs(fft_shift))
        
        # Normalize and downsample using PIL to ensure speed and stability
        mag_norm = (mag / (mag.max() + 1e-8) * 255).astype(np.uint8)
        pil_mag = Image.fromarray(mag_norm)
        pil_mag_resized = pil_mag.resize((grid_size, grid_size), Image.Resampling.BILINEAR)
        mag_resized = np.array(pil_mag_resized, dtype=np.float32) / 255.0 * mag.max()
        
        return mag_resized.tolist()
    except Exception as e:
        print("FFT 3D grid generation error:", e)
        return []

def get_visualizations(image_path: str):
    """Generate base64 visual representations of FFT and Laplacian."""
    try:
        img = Image.open(image_path).convert("L")
        w, h = img.size
        size = 256  # 256x256 is perfect for UI visualization size
        
        if w <= size or h <= size:
            crop_img = img.resize((size, size))
        else:
            x0 = (w - size) // 2
            y0 = (h - size) // 2
            crop_img = img.crop((x0, y0, x0 + size, y0 + size))
            
        arr = np.array(crop_img, dtype=np.float32)

        # 1. Centered 2D FFT Magnitude Spectrum
        fft = np.fft.fft2(arr)
        fft_shift = np.fft.fftshift(fft)
        mag = np.log1p(np.abs(fft_shift))
        mag_min, mag_max = mag.min(), mag.max()
        mag_norm = ((mag - mag_min) / (mag_max - mag_min + 1e-8) * 255).astype(np.uint8)
        
        # Colorize FFT for premium aesthetics (purple-blue glow)
        fft_colored = np.zeros((size, size, 3), dtype=np.uint8)
        fft_colored[:, :, 0] = (mag_norm * 0.4).astype(np.uint8)
        fft_colored[:, :, 1] = (mag_norm * 0.2).astype(np.uint8)
        fft_colored[:, :, 2] = (mag_norm * 0.95).astype(np.uint8)
        
        pil_fft = Image.fromarray(fft_colored)
        buffered_fft = io.BytesIO()
        pil_fft.save(buffered_fft, format="JPEG")
        fft_b64 = base64.b64encode(buffered_fft.getvalue()).decode('utf-8')

        # 2. Laplacian Edges
        lap = laplace(arr)
        lap_abs = np.abs(lap)
        lap_min, lap_max = lap_abs.min(), lap_abs.max()
        lap_norm = ((lap_abs - lap_min) / (lap_max - lap_min + 1e-8) * 255).astype(np.uint8)
        
        # Colorize edges (teal glow)
        lap_colored = np.zeros((size, size, 3), dtype=np.uint8)
        lap_colored[:, :, 0] = (lap_norm * 0.1).astype(np.uint8)
        lap_colored[:, :, 1] = (lap_norm * 0.8).astype(np.uint8)
        lap_colored[:, :, 2] = (lap_norm * 0.7).astype(np.uint8)
        
        pil_lap = Image.fromarray(lap_colored)
        buffered_lap = io.BytesIO()
        pil_lap.save(buffered_lap, format="JPEG")
        lap_b64 = base64.b64encode(buffered_lap.getvalue()).decode('utf-8')

        return fft_b64, lap_b64
    except Exception as e:
        print("Visualization generation error:", e)
        return "", ""


HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SalesCode AI - Spot the Fake Photo 3D Dashboard</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #03070c;
            --card-bg: rgba(13, 20, 35, 0.45);
            --border-color: rgba(99, 102, 241, 0.12);
            --border-glow: rgba(99, 102, 241, 0.08);
            --primary: #6366f1;
            --primary-glow: rgba(99, 102, 241, 0.35);
            --real-color: #059669;
            --real-glow: rgba(5, 150, 105, 0.15);
            --fake-color: #e11d48;
            --fake-glow: rgba(225, 29, 72, 0.15);
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', sans-serif;
        }

        body {
            background-color: transparent;
            color: var(--text-main);
            min-height: 100vh;
            padding: 2.5rem 2rem;
            position: relative;
            overflow-x: hidden;
            display: flex;
            justify-content: center;
        }

        /* 3D background canvas */
        #bg-3d-canvas {
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            z-index: -1;
            pointer-events: none;
            background-color: var(--bg-color);
        }

        .container {
            max-width: 1240px;
            width: 100%;
            display: flex;
            flex-direction: column;
            gap: 1.75rem;
            z-index: 1;
        }

        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border-color);
            padding-bottom: 1.5rem;
            position: relative;
        }

        .logo-section h1 {
            font-size: 2.2rem;
            font-weight: 700;
            letter-spacing: -0.5px;
            background: linear-gradient(135deg, #c7d2fe 0%, #6366f1 50%, #4338ca 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-shadow: 0 0 40px rgba(99, 102, 241, 0.1);
        }

        .logo-section p {
            color: var(--text-muted);
            font-size: 0.95rem;
            margin-top: 0.35rem;
        }

        .dashboard-grid {
            display: grid;
            grid-template-columns: 1.25fr 1.75fr;
            gap: 1.75rem;
        }

        @media (max-width: 1024px) {
            .dashboard-grid {
                grid-template-columns: 1fr;
            }
        }

        /* 3D Glassmorphic Cards */
        .card {
            background: var(--card-bg);
            backdrop-filter: blur(25px);
            -webkit-backdrop-filter: blur(25px);
            border: 1px solid var(--border-color);
            border-radius: 24px;
            padding: 1.75rem;
            display: flex;
            flex-direction: column;
            gap: 1.5rem;
            box-shadow: 
                0 15px 35px rgba(0, 0, 0, 0.5),
                inset 0 1px 0 rgba(255, 255, 255, 0.05),
                0 0 30px var(--border-glow);
            transform-style: preserve-3d;
            transform: perspective(1000px);
            transition: box-shadow 0.3s ease;
        }

        .card:hover {
            box-shadow: 
                0 25px 50px rgba(0, 0, 0, 0.6),
                inset 0 1px 0 rgba(255, 255, 255, 0.08),
                0 0 40px rgba(99, 102, 241, 0.15);
        }

        .card-title {
            font-size: 1.2rem;
            font-weight: 600;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            padding-bottom: 0.85rem;
            transform: translateZ(20px);
        }

        .tabs {
            display: flex;
            background: rgba(0, 0, 0, 0.4);
            border-radius: 12px;
            padding: 0.3rem;
            border: 1px solid rgba(255, 255, 255, 0.05);
            transform: translateZ(15px);
        }

        .tab-btn {
            flex: 1;
            padding: 0.6rem;
            border: none;
            background: transparent;
            color: var(--text-muted);
            font-weight: 600;
            font-size: 0.85rem;
            border-radius: 9px;
            cursor: pointer;
            transition: all 0.2s ease;
        }

        .tab-btn.active {
            background: var(--primary);
            color: white;
            box-shadow: 0 4px 15px var(--primary-glow);
        }

        .viewer-box {
            position: relative;
            width: 100%;
            aspect-ratio: 4/3;
            background: #020408;
            border-radius: 16px;
            overflow: hidden;
            border: 1px solid var(--border-color);
            display: flex;
            align-items: center;
            justify-content: center;
            transform: translateZ(10px);
        }

        video {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }

        .drop-zone {
            width: 100%;
            height: 100%;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 1rem;
            color: var(--text-muted);
            cursor: pointer;
            padding: 1.5rem;
            text-align: center;
        }

        .drop-zone svg {
            width: 56px;
            height: 56px;
            stroke: var(--text-muted);
            opacity: 0.5;
            transition: all 0.3s ease;
        }

        .drop-zone:hover svg {
            transform: scale(1.1) translateY(-4px);
            stroke: var(--primary);
            opacity: 1;
        }

        .preview-img {
            width: 100%;
            height: 100%;
            position: absolute;
            top: 0;
            left: 0;
            object-fit: cover;
            display: none;
            z-index: 10;
        }

        /* Buttons with 3D press effect */
        .btn-group {
            display: flex;
            gap: 0.85rem;
            transform: translateZ(15px);
        }

        button.action-btn {
            flex: 1;
            padding: 0.95rem;
            border: none;
            border-radius: 12px;
            font-weight: 600;
            font-size: 0.95rem;
            cursor: pointer;
            position: relative;
            overflow: hidden;
            transition: all 0.2s ease;
        }

        .btn-primary {
            background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
            color: white;
            box-shadow: 0 4px 15px var(--primary-glow);
            border-bottom: 3px solid #3730a3;
        }

        .btn-primary:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px var(--primary-glow);
        }

        .btn-primary:active {
            transform: translateY(1px);
            border-bottom-width: 0px;
        }

        .btn-secondary {
            background: rgba(255, 255, 255, 0.03);
            color: var(--text-main);
            border: 1px solid var(--border-color);
            border-bottom: 3px solid rgba(0, 0, 0, 0.3);
        }

        .btn-secondary:hover {
            background: rgba(255, 255, 255, 0.06);
            transform: translateY(-1px);
        }

        .btn-secondary:active {
            transform: translateY(1px);
            border-bottom-width: 0px;
        }

        /* Results Panel */
        .verdict-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: rgba(0, 0, 0, 0.3);
            padding: 1.25rem;
            border-radius: 16px;
            border: 1px solid var(--border-color);
            transform: translateZ(25px);
        }

        .verdict-score-wrapper {
            display: flex;
            flex-direction: column;
        }

        .score-val {
            font-size: 2.5rem;
            font-family: 'JetBrains Mono', monospace;
            font-weight: 700;
            line-height: 1.1;
            letter-spacing: -1px;
        }

        .verdict-badge {
            font-size: 1.15rem;
            font-weight: 800;
            padding: 0.6rem 1.5rem;
            border-radius: 10px;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }

        .verdict-real {
            background: var(--real-glow);
            color: #10b981;
            border: 1px solid rgba(16, 185, 129, 0.3);
            box-shadow: 0 0 20px rgba(16, 185, 129, 0.1);
        }

        .verdict-fake {
            background: var(--fake-glow);
            color: #f43f5e;
            border: 1px solid rgba(244, 63, 94, 0.3);
            box-shadow: 0 0 20px rgba(244, 63, 94, 0.1);
        }

        /* 3D Threshold Slider */
        .slider-box {
            background: rgba(0, 0, 0, 0.2);
            padding: 1.25rem;
            border-radius: 16px;
            border: 1px solid var(--border-color);
            display: flex;
            flex-direction: column;
            gap: 0.65rem;
            transform: translateZ(20px);
        }

        .slider-header {
            display: flex;
            justify-content: space-between;
            font-size: 0.85rem;
            color: var(--text-muted);
            font-weight: 600;
        }

        .slider-wrapper {
            display: flex;
            align-items: center;
            gap: 1rem;
        }

        input[type="range"] {
            flex: 1;
            height: 8px;
            background: rgba(255, 255, 255, 0.08);
            border-radius: 99px;
            outline: none;
            -webkit-appearance: none;
            cursor: pointer;
            border: 1px solid rgba(255, 255, 255, 0.03);
        }

        input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            width: 20px;
            height: 20px;
            background: var(--primary);
            border-radius: 50%;
            box-shadow: 0 0 12px var(--primary);
            border: 2px solid white;
            transition: transform 0.1s;
        }

        input[type="range"]::-webkit-slider-thumb:hover {
            transform: scale(1.2);
        }

        /* 3D interactive FFT Container */
        .viz-tab-header {
            display: flex;
            background: rgba(0, 0, 0, 0.3);
            border-radius: 10px;
            padding: 0.2rem;
            border: 1px solid rgba(255, 255, 255, 0.05);
            margin-bottom: 0.5rem;
        }

        .viz-tab-btn {
            flex: 1;
            padding: 0.4rem;
            border: none;
            background: transparent;
            color: var(--text-muted);
            font-weight: 600;
            font-size: 0.75rem;
            border-radius: 7px;
            cursor: pointer;
        }

        .viz-tab-btn.active {
            background: rgba(255, 255, 255, 0.08);
            color: white;
        }

        .fft-3d-wrapper {
            width: 100%;
            height: 240px;
            background: #020408;
            border-radius: 12px;
            border: 1px solid var(--border-color);
            position: relative;
            overflow: hidden;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        #fft-3d-canvas-container {
            width: 100%;
            height: 100%;
            cursor: grab;
        }

        #fft-3d-canvas-container:active {
            cursor: grabbing;
        }

        .viz-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.25rem;
            transform: translateZ(15px);
        }

        .viz-card {
            background: rgba(0, 0, 0, 0.2);
            border-radius: 16px;
            border: 1px solid var(--border-color);
            padding: 1rem;
            display: flex;
            flex-direction: column;
            gap: 0.75rem;
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-muted);
        }

        .viz-img-box {
            width: 100%;
            aspect-ratio: 1.2;
            background: #020408;
            border-radius: 10px;
            overflow: hidden;
            display: flex;
            align-items: center;
            justify-content: center;
            border: 1px solid rgba(255, 255, 255, 0.02);
            position: relative;
        }

        .viz-img-box img {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }

        .viz-placeholder-icon {
            font-size: 2rem;
            opacity: 0.2;
        }

        /* Diagnostic Bars */
        .features-list {
            display: flex;
            flex-direction: column;
            gap: 0.85rem;
            transform: translateZ(15px);
        }

        .feature-row {
            display: flex;
            flex-direction: column;
            gap: 0.35rem;
        }

        .feature-meta {
            display: flex;
            justify-content: space-between;
            font-size: 0.8rem;
            font-weight: 600;
        }

        .feature-label {
            color: var(--text-main);
        }

        .feature-val {
            color: var(--text-muted);
            font-family: 'JetBrains Mono', monospace;
        }

        .progress-track {
            background: rgba(0, 0, 0, 0.4);
            height: 8px;
            border-radius: 99px;
            overflow: hidden;
            border: 1px solid rgba(255, 255, 255, 0.03);
            position: relative;
        }

        .progress-fill {
            height: 100%;
            width: 0%;
            transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
            border-radius: 99px;
            box-shadow: 0 0 10px rgba(99, 102, 241, 0.5);
        }

        /* History Logs */
        .history-list {
            display: flex;
            flex-direction: column;
            gap: 0.65rem;
            max-height: 160px;
            overflow-y: auto;
            transform: translateZ(10px);
        }

        .history-item {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: rgba(0, 0, 0, 0.2);
            padding: 0.65rem 1rem;
            border-radius: 10px;
            font-size: 0.85rem;
            border: 1px solid rgba(255, 255, 255, 0.02);
            transition: background 0.2s;
        }

        .history-item:hover {
            background: rgba(255, 255, 255, 0.02);
        }

        .history-score {
            font-family: 'JetBrains Mono', monospace;
            font-weight: 700;
        }

        /* General Loader and States */
        .results-inner {
            display: none;
            flex-direction: column;
            gap: 1.5rem;
        }

        .results-empty {
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            color: var(--text-muted);
            gap: 1rem;
            text-align: center;
            min-height: 380px;
        }

        .results-empty svg {
            width: 64px;
            height: 64px;
            stroke: var(--text-muted);
            opacity: 0.2;
        }

        .loader-box {
            display: none;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            gap: 1.25rem;
            min-height: 380px;
        }

        .loader {
            width: 40px;
            height: 40px;
            border: 4px solid rgba(255, 255, 255, 0.05);
            border-top: 4px solid var(--primary);
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }

        .pulse-text {
            color: var(--text-muted);
            font-size: 0.95rem;
            font-weight: 500;
            animation: pulse 1.5s infinite;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        @keyframes pulse {
            0% { transform: scale(1); opacity: 0.6; }
            50% { transform: scale(1.02); opacity: 1; }
            100% { transform: scale(1); opacity: 0.6; }
        }
    </style>
    <!-- Three.js + OrbitControls + VanillaTilt for 3D UI -->
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/vanilla-tilt/1.7.2/vanilla-tilt.min.js"></script>
</head>
<body>
    <!-- Background 3D Particles -->
    <div id="bg-3d-canvas"></div>

    <div class="container">
        <header>
            <div class="logo-section">
                <h1>Spot the Fake Photo 3D Dashboard</h1>
                <p>Advanced Explainable Image Authenticity Visual Diagnostic Station</p>
            </div>
            <div style="font-size: 0.85rem; text-align: right; color: var(--text-muted);">
                Model size: <span style="color: var(--text-main); font-weight: 600;">283 KB (PKL)</span><br>
                LOO-CV Accuracy: <span style="color: #10b981; font-weight: 700;">95.9%</span>
            </div>
        </header>

        <div class="dashboard-grid">
            <!-- Left Column: Acquisition & 3D visualizer -->
            <div style="display: flex; flex-direction: column; gap: 1.75rem;">
                
                <!-- Image Acquisition Card -->
                <div class="card" data-tilt data-tilt-max="4" data-tilt-speed="400" data-tilt-glare="true" data-tilt-max-glare="0.1">
                    <div class="card-title">Image Acquisition</div>
                    <div class="tabs">
                        <button id="tab-webcam" class="tab-btn active">Live Webcam</button>
                        <button id="tab-file" class="tab-btn">Upload Image File</button>
                    </div>

                    <div class="viewer-box">
                        <video id="webcam" autoplay playsinline></video>
                        
                        <div id="file-dropzone" class="drop-zone" style="display: none;">
                            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
                                <path stroke-linecap="round" stroke-linejoin="round" d="M2.25 15.75l5.159-5.159a2.25 2.25 0 013.182 0l5.159 5.159m-1.5-1.5l1.409-1.409a2.25 2.25 0 013.182 0l2.909 2.909m-18 3.75h16.5a1.5 1.5 0 001.5-1.5V6a1.5 1.5 0 00-1.5-1.5H3.75A1.5 1.5 0 002.25 6v12a1.5 1.5 0 001.5 1.5zm10.5-11.25h.008v.008h-.008V8.25zm.375 0a.375 0 11-.75 0 .375 0 01.75 0z" />
                            </svg>
                            <div>
                                <p style="font-weight: 600; font-size: 0.95rem; color: var(--text-main);">Drag & Drop Image Here</p>
                                <p style="font-size: 0.8rem; margin-top: 0.25rem;">or click to browse local files</p>
                            </div>
                            <input type="file" id="file-input" accept="image/*" style="display: none;">
                        </div>

                        <img id="preview" class="preview-img" alt="Captured acquisition">
                    </div>

                    <div class="btn-group">
                        <button id="capture-btn" class="action-btn btn-primary">Capture & Predict</button>
                        <button id="reset-btn" class="action-btn btn-secondary" style="display: none;">Reset / Retake</button>
                    </div>
                </div>

                <!-- Signal Visualization Card -->
                <div class="card" data-tilt data-tilt-max="3" data-tilt-speed="400" data-tilt-glare="true" data-tilt-max-glare="0.05">
                    <div class="card-title">
                        <span>Signal Visualizations</span>
                        <div class="viz-tab-header">
                            <button id="btn-viz-2d" class="viz-tab-btn active">2D Images</button>
                            <button id="btn-viz-3d" class="viz-tab-btn">3D FFT Mesh</button>
                        </div>
                    </div>
                    
                    <!-- 2D Image Grid Visualizer -->
                    <div class="viz-grid" id="viz-container-2d">
                        <div class="viz-card">
                            FFT Spectrum (2D)
                            <div class="viz-img-box">
                                <span class="viz-placeholder-icon" id="fft-placeholder">🌐</span>
                                <img id="fft-img" style="display: none;" alt="FFT Spectrum">
                            </div>
                        </div>
                        <div class="viz-card">
                            Laplacian Edge Map
                            <div class="viz-img-box">
                                <span class="viz-placeholder-icon" id="lap-placeholder">⚡</span>
                                <img id="lap-img" style="display: none;" alt="Laplacian edges">
                            </div>
                        </div>
                    </div>

                    <!-- 3D WebGL FFT Surface Plot -->
                    <div id="viz-container-3d" class="fft-3d-wrapper" style="display: none;">
                        <span id="fft-3d-placeholder" class="viz-placeholder-icon" style="position: absolute; pointer-events:none; z-index:5;">🌐 Awaiting 3D Mesh</span>
                        <div id="fft-3d-canvas-container"></div>
                    </div>
                </div>
            </div>

            <!-- Right Column: Results & Interactive Diagnostic Dashboard -->
            <div style="display: flex; flex-direction: column; gap: 1.75rem;">
                <div class="card" style="flex: 1;" data-tilt data-tilt-max="3" data-tilt-speed="400">
                    <div class="card-title">Authenticity Analysis Engine</div>

                    <!-- Loader Box -->
                    <div class="loader-box" id="loader-container">
                        <div class="loader"></div>
                        <div class="pulse-text">Extracting spatial and frequency features...</div>
                    </div>

                    <!-- Unanalyzed State -->
                    <div class="results-empty" id="placeholder-box">
                        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
                            <line x1="9" y1="3" x2="9" y2="21" />
                            <line x1="15" y1="3" x2="15" y2="21" />
                        </svg>
                        <p>Awaiting photo input.<br><span style="font-size: 0.85rem; color: var(--text-muted);">Acquire or upload a photo to populate the analysis dashboard.</span></p>
                    </div>

                    <!-- Analyzed State -->
                    <div class="results-inner" id="results-inner-box">
                        <div class="verdict-header">
                            <div class="verdict-score-wrapper">
                                <span style="font-size: 0.75rem; color: var(--text-muted); font-weight: 700; text-transform: uppercase;">Fake Likelihood</span>
                                <span class="score-val" id="score-text">0.00%</span>
                            </div>
                            <span id="verdict-badge" class="verdict-badge verdict-real">REAL</span>
                        </div>

                        <!-- Threshold Adjuster -->
                        <div class="slider-box">
                            <div class="slider-header">
                                <span>Adjust Decision Cut-off</span>
                                <span id="threshold-val">Threshold: 0.50</span>
                            </div>
                            <div class="slider-wrapper">
                                <span style="font-size: 0.75rem; font-weight: 700; color: #10b981;">REAL</span>
                                <input type="range" id="threshold-slider" min="0.1" max="0.9" step="0.01" value="0.50">
                                <span style="font-size: 0.75rem; font-weight: 700; color: #f43f5e;">SCREEN</span>
                            </div>
                            <span style="font-size: 0.75rem; color: var(--text-muted); text-align: center; margin-top: 0.25rem;">
                                Higher threshold reduces False Positives (real users flagged as fake) but might miss clever fakes.
                            </span>
                        </div>

                        <!-- Explainable Features Breakdown -->
                        <div style="display: flex; flex-direction: column; gap: 0.75rem;">
                            <span style="font-size: 0.85rem; font-weight: 700; color: var(--text-muted); text-transform: uppercase;">Handcrafted Feature Diagnostics</span>
                            <div class="features-list" id="features-list-container">
                                <!-- Dynamic Diagnostic Rows Injected Here -->
                            </div>
                        </div>

                        <!-- Session History Logs -->
                        <div style="display: flex; flex-direction: column; gap: 0.5rem; margin-top: 0.5rem; border-top: 1px solid rgba(255, 255, 255, 0.05); padding-top: 1rem;">
                            <span style="font-size: 0.85rem; font-weight: 700; color: var(--text-muted); text-transform: uppercase;">Acquisition Session History</span>
                            <div class="history-list" id="history-container">
                                <p style="font-size: 0.8rem; color: var(--text-muted); font-style: italic; text-align: center; padding: 0.5rem 0;">No tests recorded yet.</p>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <canvas id="canvas"></canvas>

    <script>
        const video = document.getElementById('webcam');
        const canvas = document.getElementById('canvas');
        const captureBtn = document.getElementById('capture-btn');
        const resetBtn = document.getElementById('reset-btn');
        const preview = document.getElementById('preview');
        
        const placeholderBox = document.getElementById('placeholder-box');
        const resultsInnerBox = document.getElementById('results-inner-box');
        const loaderContainer = document.getElementById('loader-container');
        
        const scoreText = document.getElementById('score-text');
        const verdictBadge = document.getElementById('verdict-badge');
        const featuresContainer = document.getElementById('features-list-container');
        const historyContainer = document.getElementById('history-container');
        
        const tabWebcam = document.getElementById('tab-webcam');
        const tabFile = document.getElementById('tab-file');
        const fileDropzone = document.getElementById('file-dropzone');
        const fileInput = document.getElementById('file-input');

        const fftImg = document.getElementById('fft-img');
        const lapImg = document.getElementById('lap-img');
        const fftPlaceholder = document.getElementById('fft-placeholder');
        const lapPlaceholder = document.getElementById('lap-placeholder');

        const thresholdSlider = document.getElementById('threshold-slider');
        const thresholdVal = document.getElementById('threshold-val');

        const btnViz2D = document.getElementById('btn-viz-2d');
        const btnViz3D = document.getElementById('btn-viz-3d');
        const vizContainer2D = document.getElementById('viz-container-2d');
        const vizContainer3D = document.getElementById('viz-container-3d');
        const fft3dPlaceholder = document.getElementById('fft-3d-placeholder');

        let activeTab = 'webcam';
        let activeVizTab = '2d';
        let selectedFileBase64 = null;
        let stream = null;
        
        let lastScore = null;
        let historyLogs = [];

        // Three.js instances
        let bgScene, bgCamera, bgRenderer, starGeo, stars;
        let scene, camera, renderer, controls, fftMesh;

        // UI Metadata
        const featureMetadata = {
            "noise_level": { label: "Sensor Noise Grain (Smooth Areas)", color: "#3b82f6", desc: "Low value is highly indicative of screens." },
            "laplacian_variance": { label: "Edge Sharpness (Laplacian Var)", color: "#10b981", desc: "Var in edges. Recaptures tend to have higher/lower focus spreads." },
            "jpeg_blockiness": { label: "JPEG 8x8 DCT Compression Artifacts", color: "#f59e0b", desc: "Measures grid blockiness from double compression." },
            "brightness_mean": { label: "Mean Illumination (Backlight Glow)", color: "#8b5cf6", desc: "High brightness glow from monitor/laptop screen backlight." },
            "hf_energy_ratio": { label: "FFT High Frequency Power Ratio", color: "#ec4899", desc: "High energy = screen grid Moiré signals." },
            "directional_asymmetry": { label: "Axis-Aligned H/V Grid Asymmetry", color: "#06b6d4", desc: "Grid lines in screens align along horizontal/vertical axes." },
            "hf_peak_to_mean": { label: "FFT Spectral Peak-to-Mean Ratio", color: "#f43f5e", desc: "Screens show sharp localized peak spikes." },
            "saturation_entropy": { label: "HSV Saturation Histogram Entropy", color: "#a855f7", desc: "Screens compress color ranges, shifting saturation entropy." },
            "brightness_std": { label: "Illumination Standard Deviation", color: "#64748b", desc: "High variation = natural lighting patterns." },
            "glare_ratio": { label: "Surface Specular Glare Coverage", color: "#e2e8f0", desc: "Identifies glare reflections on monitor glass." },
            "chroma_blur_ratio": { label: "Color Fringing Sharpness Variance", color: "#0ea5e9", desc: "Real lenses cause R/G/B sharpness dispersion. Screens don't." },
            "saturation_std": { label: "Color Saturation Standard Dev", color: "#f97316", desc: "Uniformity of saturation. Natural objects vary more than screens." },
            "fft_radial_falloff": { label: "FFT Spectral Radial Falloff Slope", color: "#14b8a6", desc: "Radial falloff rate. Natural is smooth, screens have grid bumps." }
        };

        // ── 3D Background Stars setup ───────────────────────────────────────
        function initBg3D() {
            const bgContainer = document.getElementById('bg-3d-canvas');
            if (!bgContainer) return;
            
            bgScene = new THREE.Scene();
            bgCamera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 1, 1000);
            bgCamera.position.z = 1;
            bgCamera.rotation.x = Math.PI / 2;
            
            bgRenderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
            bgRenderer.setSize(window.innerWidth, window.innerHeight);
            bgContainer.appendChild(bgRenderer.domElement);
            
            starGeo = new THREE.BufferGeometry();
            const starCount = 800;
            const positions = [];
            
            for(let i=0; i<starCount; i++) {
                positions.push(
                    Math.random() * 600 - 300,
                    Math.random() * 600 - 300,
                    Math.random() * 600 - 300
                );
            }
            starGeo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
            
            // Texture for glow
            const pCanvas = document.createElement('canvas');
            pCanvas.width = 16;
            pCanvas.height = 16;
            const pCtx = pCanvas.getContext('2d');
            const grad = pCtx.createRadialGradient(8, 8, 0, 8, 8, 8);
            grad.addColorStop(0, 'rgba(255, 255, 255, 1)');
            grad.addColorStop(1, 'rgba(99, 102, 241, 0)');
            pCtx.fillStyle = grad;
            pCtx.fillRect(0, 0, 16, 16);
            const pTexture = new THREE.CanvasTexture(pCanvas);
            
            const starMaterial = new THREE.PointsMaterial({
                color: 0x818cf8,
                size: 2.5,
                map: pTexture,
                transparent: true,
                opacity: 0.5,
                depthWrite: false
            });
            
            stars = new THREE.Points(starGeo, starMaterial);
            bgScene.add(stars);
            
            window.addEventListener('resize', onWindowResize);
            document.addEventListener('mousemove', onDocumentMouseMove);
            
            animateBg();
        }

        let mouseX = 0, mouseY = 0;
        function onDocumentMouseMove(e) {
            mouseX = (e.clientX - window.innerWidth / 2) * 0.05;
            mouseY = (e.clientY - window.innerHeight / 2) * 0.05;
        }

        function onWindowResize() {
            bgCamera.aspect = window.innerWidth / window.innerHeight;
            bgCamera.updateProjectionMatrix();
            bgRenderer.setSize(window.innerWidth, window.innerHeight);
        }

        function animateBg() {
            requestAnimationFrame(animateBg);
            const positions = starGeo.attributes.position.array;
            for(let i=1; i<positions.length; i+=3) {
                positions[i] -= 0.15;
                if (positions[i] < -300) {
                    positions[i] = 300;
                }
            }
            starGeo.attributes.position.needsUpdate = true;
            stars.rotation.y += 0.0004;
            bgScene.rotation.x += (mouseY * 0.0005 - bgScene.rotation.x) * 0.05;
            bgScene.rotation.y += (mouseX * 0.0005 - bgScene.rotation.y) * 0.05;
            bgRenderer.render(bgScene, bgCamera);
        }

        // ── 3D FFT Mesh Visualizer setup ────────────────────────────────────
        const container = document.getElementById('fft-3d-canvas-container');

        function init3DFFT() {
            scene = new THREE.Scene();
            camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 1, 1000);
            camera.position.set(130, 110, 130);

            renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
            renderer.setSize(container.clientWidth, container.clientHeight);
            renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
            container.appendChild(renderer.domElement);

            controls = new THREE.OrbitControls(camera, renderer.domElement);
            controls.enableDamping = true;
            controls.dampingFactor = 0.05;
            controls.maxPolarAngle = Math.PI / 2 - 0.05;

            // Lights
            scene.add(new THREE.AmbientLight(0xffffff, 0.45));
            const dLight = new THREE.DirectionalLight(0x6366f1, 1.2);
            dLight.position.set(100, 150, 50);
            scene.add(dLight);
            const pLight = new THREE.PointLight(0xf43f5e, 1.5, 300);
            pLight.position.set(-100, 100, -50);
            scene.add(pLight);

            // Plane Mesh (32x32 vertices)
            const geometry = new THREE.PlaneGeometry(140, 140, 31, 31);
            geometry.rotateX(-Math.PI / 2);

            const colors = [];
            const count = geometry.attributes.position.count;
            for (let i = 0; i < count; i++) {
                colors.push(0.388, 0.4, 0.945); // default indigo
            }
            geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

            const material = new THREE.MeshStandardMaterial({
                color: 0xffffff,
                vertexColors: true,
                wireframe: true,
                transparent: true,
                opacity: 0.85,
                roughness: 0.2,
                metalness: 0.1
            });

            fftMesh = new THREE.Mesh(geometry, material);
            scene.add(fftMesh);

            animate3DFFT();
        }

        function animate3DFFT() {
            requestAnimationFrame(animate3DFFT);
            controls.update();
            if (fftMesh) {
                fftMesh.rotation.y += 0.001;
            }
            renderer.render(scene, camera);
        }

        function update3DFFTMesh(grid) {
            if (!fftMesh || !grid || grid.length === 0) return;
            const geometry = fftMesh.geometry;
            const position = geometry.attributes.position;
            const colorsAttr = geometry.attributes.color;

            let idx = 0;
            let maxVal = 0.01;
            for (let y = 0; y < 32; y++) {
                for (let x = 0; x < 32; x++) {
                    if (grid[y][x] > maxVal) maxVal = grid[y][x];
                }
            }

            for (let y = 0; y < 32; y++) {
                for (let x = 0; x < 32; x++) {
                    const val = grid[y][x];
                    const height = val * 5.5; 
                    position.setY(idx, height - 12.0); // center on geometry Y plane

                    const ratio = val / maxVal;
                    // Interpolate from deep indigo to hot pink/cyan
                    const r = 0.15 + ratio * 0.8;
                    const g = 0.15 + ratio * 0.2;
                    const b = 0.45 + ratio * 0.5;
                    colorsAttr.setXYZ(idx, r, g, b);
                    idx++;
                }
            }
            position.needsUpdate = true;
            colorsAttr.needsUpdate = true;
            geometry.computeVertexNormals();
        }

        // Start Webcam stream
        async function startWebcam() {
            try {
                if (stream) {
                    stream.getTracks().forEach(track => track.stop());
                }
                stream = await navigator.mediaDevices.getUserMedia({
                    video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 720 } }
                });
                video.srcObject = stream;
                video.style.display = 'block';
                fileDropzone.style.display = 'none';
            } catch (err) {
                console.error("Camera error:", err);
                switchToFileTab();
            }
        }

        function switchToFileTab() {
            activeTab = 'file';
            tabFile.classList.add('active');
            tabWebcam.classList.remove('active');
            if (stream) {
                stream.getTracks().forEach(track => track.stop());
                stream = null;
            }
            video.style.display = 'none';
            fileDropzone.style.display = 'flex';
            captureBtn.innerText = 'Run Analysis';
            resetDemo();
        }

        tabWebcam.addEventListener('click', () => {
            activeTab = 'webcam';
            tabWebcam.classList.add('active');
            tabFile.classList.remove('active');
            captureBtn.innerText = 'Capture & Predict';
            preview.style.display = 'none';
            resetDemo();
            startWebcam();
        });

        tabFile.addEventListener('click', switchToFileTab);

        fileDropzone.addEventListener('click', () => fileInput.click());
        fileInput.addEventListener('change', handleFileSelect);

        // Drag & Drop events
        fileDropzone.addEventListener('dragover', (e) => { e.preventDefault(); fileDropzone.style.background = 'rgba(255, 255, 255, 0.04)'; });
        fileDropzone.addEventListener('dragleave', () => { fileDropzone.style.background = 'transparent'; });
        fileDropzone.addEventListener('drop', (e) => {
            e.preventDefault();
            fileDropzone.style.background = 'transparent';
            if (e.dataTransfer.files.length > 0) {
                fileInput.files = e.dataTransfer.files;
                handleFileSelect();
            }
        });

        function handleFileSelect() {
            const file = fileInput.files[0];
            if (!file) return;
            const reader = new FileReader();
            reader.onload = function(e) {
                selectedFileBase64 = e.target.result;
                preview.src = selectedFileBase64;
                preview.style.display = 'block';
                fileDropzone.style.display = 'none';
                resetBtn.style.display = 'block';
            };
            reader.readAsDataURL(file);
        }

        // Toggle visualizer tabs
        btnViz2D.addEventListener('click', () => {
            activeVizTab = '2d';
            btnViz2D.classList.add('active');
            btnViz3D.classList.remove('active');
            vizContainer2D.style.display = 'grid';
            vizContainer3D.style.display = 'none';
        });

        btnViz3D.addEventListener('click', () => {
            activeVizTab = '3d';
            btnViz3D.classList.add('active');
            btnViz2D.classList.remove('active');
            vizContainer3D.style.display = 'flex';
            vizContainer2D.style.display = 'none';
        });

        // Trigger analysis click
        captureBtn.addEventListener('click', async () => {
            let dataUrl = null;
            let sourceName = "";

            if (activeTab === 'webcam') {
                const width = video.videoWidth;
                const height = video.videoHeight;
                if (!width || !height) return;

                canvas.width = width;
                canvas.height = height;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(video, 0, 0, width, height);
                dataUrl = canvas.toDataURL('image/jpeg', 0.95);
                preview.src = dataUrl;
                preview.style.display = 'block';
                sourceName = "Webcam Capture";
            } else {
                if (!selectedFileBase64) {
                    alert("Please select or drop a photo file first!");
                    return;
                }
                dataUrl = selectedFileBase64;
                sourceName = fileInput.files[0] ? fileInput.files[0].name : "Uploaded File";
            }

            resetBtn.style.display = 'block';
            captureBtn.style.display = 'none';
            placeholderBox.style.display = 'none';
            resultsInnerBox.style.display = 'none';
            loaderContainer.style.display = 'flex';

            fftImg.style.display = 'none';
            lapImg.style.display = 'none';
            fftPlaceholder.style.display = 'inline';
            lapPlaceholder.style.display = 'inline';
            fft3dPlaceholder.style.display = 'inline';

            try {
                const response = await fetch('/predict_dashboard', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image: dataUrl.split(',')[1] })
                });
                const resData = await response.json();
                
                loaderContainer.style.display = 'none';
                resultsInnerBox.style.display = 'flex';

                if (resData.error) {
                    alert(resData.error);
                    resetDemo();
                    return;
                }

                // Render 2D visualizations
                if (resData.fft_base64 && resData.laplacian_base64) {
                    fftImg.src = "data:image/jpeg;base64," + resData.fft_base64;
                    lapImg.src = "data:image/jpeg;base64," + resData.laplacian_base64;
                    fftImg.style.display = 'block';
                    lapImg.style.display = 'block';
                    fftPlaceholder.style.display = 'none';
                    lapPlaceholder.style.display = 'none';
                }

                // Render 3D terrain
                if (resData.fft_3d_grid && resData.fft_3d_grid.length > 0) {
                    update3DFFTMesh(resData.fft_3d_grid);
                    fft3dPlaceholder.style.display = 'none';
                }

                // Render score & verdict
                lastScore = resData.score;
                updateVerdictDisplay();

                // Render diagnostics bars
                featuresContainer.innerHTML = "";
                Object.keys(resData.features).forEach(fname => {
                    const rawVal = resData.features[fname];
                    const meta = featureMetadata[fname] || { label: fname, color: "#6366f1", desc: "" };
                    
                    let dispPct = 50; 
                    if (fname === "noise_level") dispPct = Math.min(Math.max((rawVal / 15) * 100, 5), 95);
                    else if (fname === "laplacian_variance") dispPct = Math.min(Math.max((rawVal / 2500) * 100, 5), 95);
                    else if (fname === "jpeg_blockiness") dispPct = Math.min(Math.max((rawVal - 0.7) * 150, 5), 95);
                    else if (fname === "brightness_mean") dispPct = Math.min(Math.max(rawVal * 100, 5), 95);
                    else if (fname === "hf_energy_ratio") dispPct = Math.min(Math.max((rawVal - 0.8) * 500, 5), 95);
                    else dispPct = Math.min(Math.max(rawVal * 100, 5), 95);
                    
                    const rowHtml = `
                        <div class="feature-row" title="${meta.desc}">
                            <div class="feature-meta">
                                <span class="feature-label">${meta.label}</span>
                                <span class="feature-val">${rawVal.toFixed(4)}</span>
                            </div>
                            <div class="progress-track">
                                <div class="progress-fill" style="width: ${dispPct}%; background-color: ${meta.color};"></div>
                            </div>
                        </div>
                    `;
                    featuresContainer.insertAdjacentHTML('beforeend', rowHtml);
                });

                // Add to history
                const verdict = lastScore >= thresholdSlider.value ? "SCREEN" : "REAL";
                historyLogs.unshift({
                    source: sourceName,
                    score: lastScore,
                    verdict: verdict,
                    time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
                });
                renderHistory();

            } catch (err) {
                console.error("Dashboard Predict Error:", err);
                alert("Failed to analyze image.");
                resetDemo();
            }
        });

        // Threshold Slider event
        thresholdSlider.addEventListener('input', () => {
            const thresh = parseFloat(thresholdSlider.value).toFixed(2);
            thresholdVal.innerText = "Threshold: " + thresh;
            updateVerdictDisplay();
        });

        function updateVerdictDisplay() {
            if (lastScore === null) return;
            const scorePct = (lastScore * 100).toFixed(1) + '%';
            scoreText.innerText = scorePct;

            const currentThreshold = parseFloat(thresholdSlider.value);
            if (lastScore >= currentThreshold) {
                verdictBadge.innerText = 'SCREEN (fake)';
                verdictBadge.className = 'verdict-badge verdict-fake';
                scoreText.style.color = 'var(--fake-color)';
            } else {
                verdictBadge.innerText = 'REAL';
                verdictBadge.className = 'verdict-badge verdict-real';
                scoreText.style.color = 'var(--real-color)';
            }
        }

        function renderHistory() {
            if (historyLogs.length === 0) {
                historyContainer.innerHTML = '<p style="font-size: 0.8rem; color: var(--text-muted); font-style: italic; text-align: center; padding: 0.5rem 0;">No tests recorded yet.</p>';
                return;
            }
            historyContainer.innerHTML = "";
            historyLogs.slice(0, 4).forEach(item => {
                const color = item.score >= parseFloat(thresholdSlider.value) ? 'var(--fake-color)' : 'var(--real-color)';
                const label = item.score >= parseFloat(thresholdSlider.value) ? 'SCREEN' : 'REAL';
                
                const itemHtml = `
                    <div class="history-item">
                        <div style="display:flex; flex-direction:column; gap:0.15rem;">
                            <span style="font-weight: 600; overflow: hidden; text-overflow: ellipsis; max-width: 170px; white-space: nowrap;">${item.source}</span>
                            <span style="font-size:0.7rem; color:var(--text-muted);">${item.time}</span>
                        </div>
                        <div style="text-align: right; display:flex; flex-direction:column; gap:0.15rem;">
                            <span class="history-score" style="color: ${color};">${(item.score * 100).toFixed(1)}%</span>
                            <span style="font-size:0.65rem; color: var(--text-muted); font-weight:600; text-transform:uppercase;">${label}</span>
                        </div>
                    </div>
                `;
                historyContainer.insertAdjacentHTML('beforeend', itemHtml);
            });
        }

        resetBtn.addEventListener('click', resetDemo);

        function resetDemo() {
            preview.style.display = 'none';
            resetBtn.style.display = 'none';
            captureBtn.style.display = 'block';
            placeholderBox.style.display = 'flex';
            resultsInnerBox.style.display = 'none';
            loaderContainer.style.display = 'none';
            selectedFileBase64 = null;
            lastScore = null;
            if (activeTab === 'file') {
                fileDropzone.style.display = 'flex';
                fileInput.value = '';
            }
            fftImg.style.display = 'none';
            lapImg.style.display = 'none';
            fftPlaceholder.style.display = 'inline';
            lapPlaceholder.style.display = 'inline';
            fft3dPlaceholder.style.display = 'inline';
            
            // Reset 3D mesh
            if (fftMesh) {
                const geometry = fftMesh.geometry;
                const position = geometry.attributes.position;
                const colorsAttr = geometry.attributes.color;
                const count = position.count;
                for (let i = 0; i < count; i++) {
                    position.setY(i, -10.0);
                    colorsAttr.setXYZ(i, 0.388, 0.4, 0.945);
                }
                position.needsUpdate = true;
                colorsAttr.needsUpdate = true;
                geometry.computeVertexNormals();
            }
        }

        // Initialize elements
        initBg3D();
        init3DFFT();
        startWebcam();
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict_dashboard', methods=['POST'])
def predict_dashboard():
    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({'error': 'No image data provided'}), 400

    try:
        # Decode base64 string to a temporary file
        img_bytes = base64.b64decode(data['image'])
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            f.write(img_bytes)
            tmp_path = f.name

        # 1. Run inference
        start = time.perf_counter()
        score = predict(tmp_path)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        # 2. Extract raw features dictionary for display
        raw_feats = extract_features(tmp_path)
        feature_dict = {}
        for name, val in zip(FEATURE_NAMES, raw_feats):
            feature_dict[name] = float(val)

        # 3. Generate base64 representations of FFT and Laplacian
        fft_b64, lap_b64 = get_visualizations(tmp_path)

        # 4. Generate 32x32 FFT grid data for 3D visualizer
        fft_3d_grid = get_fft_3d_grid(tmp_path, grid_size=32)

        # Clean up
        try:
            os.remove(tmp_path)
        except Exception:
            pass

        return jsonify({
            'score': score,
            'features': feature_dict,
            'fft_base64': fft_b64,
            'laplacian_base64': lap_b64,
            'fft_3d_grid': fft_3d_grid,
            'latency_ms': elapsed_ms
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("*" * 60)
    print("SalesCode AI - Spot the Fake Photo 3D Dashboard starting...")
    print("Open http://localhost:5000 in your browser to test live!")
    print("*" * 60)
    app.run(host='0.0.0.0', port=5000, debug=False)
