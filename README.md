# Spot the Fake Photo — Explainable Image Authenticity Station

A lightweight, high-performance photo-authenticity detector that distinguishes **real photographs** from **photos taken of a screen** (recaptured screen fraud/screenshot cheating).

 **LOO-CV Accuracy**: **95.9%**  
 **Average Latency**: **~514 ms** (CPU-only, no GPU required)  
 **On-device Cost**: **$0.00 per image** (100% local, no API calls, no cloud servers)  
 **Model footprint**: **283 KB** (Random Forest classifier + JSON metadata)

---

##  3D Visual Diagnostics Dashboard (Live Demo)

The project includes an **interactive 3D WebGL Dashboard** served locally using Flask, Three.js, and Vanilla-Tilt.js.

* **3D FFT Terrain Mesh**: Projects the 2D spatial frequency spectrum onto a rotatable 3D mesh. Natural images show a single smooth central cone; screen recaptures display prominent spiky wireframe peaks corresponding to the pixel grids.
* **3D Particle Background & Card Tilt**: High-fidelity visual aesthetics that tilt in perspective space on mouse movement.
* **Diagnostics Panel**: Visualizes all 13 feature metrics on active progress gauges.
* **Decision Slider**: Interactive cutoff threshold slider lets you test precision/recall boundaries real-time.

### How to Run the Dashboard:
```bash
python app.py
```
Open **`http://localhost:5000`** in your browser. (If testing on a mobile device on the same network, open the custom IP served, and use the **Upload** tab fallback, bypassing camera security constraints).

---

##  Submission Note (Half-Page Summary)

### Approach
I built a classical signal-processing pipeline that reduces each image to 13 handcrafted features (FFT frequency analysis for pixel-grids, JPEG blockiness for re-encoding artifacts, noise level in smooth regions, Laplacian variance for sharpness, HSV statistics, chromatic aberration ratio, and glare ratio). These features are passed to a Random Forest classifier (selected via Leave-One-Out CV out of 9 candidates). To keep latency low, I implemented PIL native cropping: the image is loaded lazily, and only five 512×512 patches are extracted and converted to NumPy for FFT/spatial analysis.

### Why This Approach
It satisfies the "small, fast, cheap, honest" criteria. By lazy-loading and cropping in PIL space before NumPy conversions, I cut memory allocations by 95% and reduced CPU latency from 1.7s to ~500ms. FFT is performed on small 512px patches, preserving high-frequency moiré patterns that would be destroyed by downsampling the entire image. The model runs completely on-device for free ($0 cloud cost).

### Honest Accuracy
**LOO-CV accuracy: 95.9%** on my 97-image dataset (47 real, 50 screen).
I used Leave-One-Out CV to prevent train/test leakage on a small sample size. The final model resulted in only 1 false positive (real photo flagged as fake) and 3 false negatives (screen recaptures missed). 

### What I'd Improve With More Time
1. **GPU-accelerated FFT**: Porting patch FFTs to a mobile GPU (via CoreML/Metal or WebGL) would drop latency from 500ms to <10ms.
2. **More diverse data**: Train with different screen technologies (e.g., OLED, anti-glare, curved), outdoor/indoor lighting variations, and printed paper recaptures.
3. **Double-JPEG tuning**: Make JPEG blockiness alignment-invariant to catch screens regardless of camera crop angle.

---

## Feature Engineering (13 Diagnostic Signals)

| Feature | Category | Purpose |
|---|---|---|
| `noise_level` | Spatial | Measures sensor ISO noise grain in flat areas (screens render solid pixels with near-zero noise) |
| `laplacian_variance` | Spatial | Measures edge sharpness (sharp grid patterns vs natural depth of field) |
| `jpeg_blockiness` | Compression | Measures double-JPEG 8×8 DCT block boundaries created by screen re-encoding |
| `chroma_blur_ratio` | Spatial | Measures color fringing (chromatic aberration). Lenses create color shifts; screens display perfect pixel-aligned RGB |
| `saturation_std` | Color | Measures uniformity of saturation (screens compress color depth, making saturation more uniform) |
| `fft_radial_falloff` | Frequency | Measures spectral falloff rate (natural textures drop off smoothly; pixel grids create high-freq bumps) |
| `hf_peak_to_mean` | Frequency | Measures moiré frequency peak spikes in the FFT magnitude |
| `hf_energy_ratio` | Frequency | Measures global high-frequency energy ratio |
| `directional_asymmetry` | Frequency | Measures axis-aligned grid orientation (screens show grid asymmetry, natural images are isotropic) |
| `brightness_mean` | Color | Measures screen glow / monitor backlight levels |
| `brightness_std` | Color | Measures natural scene lighting variations |
| `saturation_entropy` | Color | Measures distribution shape of saturation |
| `glare_ratio` | Spatial | Detects specular screen glare hotspots |

---

##  CLI Setup & Usage

### Setup
```bash
pip install -r requirements.txt
```

### Run Inference
```bash
python predict.py some_image.jpg
```
**Output Contract**: Returns a score in `[0.0, 1.0]` where `0` is a certain REAL photo and `1` is a certain SCREEN recapture.

### Evaluate Dataset
```bash
python evaluate.py
```
Re-runs cross-validation and generates confusion matrices, classification reports, and recommended thresholds.

### Latency Benchmark
```bash
python benchmark.py
```
Generates average, median, and P95 latency measurements alongside throughput and scaling cost audits.
