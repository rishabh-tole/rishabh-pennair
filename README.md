# Penn Air Software Application 2025

This project implements a shape detection pipeline that started from simple masking and evolved to an **optical-flow-based** method with **contour refinement** and a **3D extension** using basic pinhole optics.

- **Static images:** Detect and overlay shapes/centers  
- **Dynamic videos:** Robust to backgrounds & colors via pruning/refinement  
- **3D (meters):** Estimate (X, Y, Z) of detected shapes using intrinsics + known circle size  
- **ROS2:** Not completed

---

## 🔧 Installation

> Requires Python 3.8+.

```bash
# 1) Clone and enter the repo
git clone <YOUR-REPO-URL>
cd <YOUR-REPO-FOLDER>

# 2) Install dependencies
pip install -r requirements.txt
```

---

## 📁 Files

- **shape_detector.py** — Core class: optical flow (forward diff), tolerant zero-map, neighbor pruning, contour drawing.  
- **part1.py** — Static image pipeline; quick visual overlay (Matplotlib).  
- **part2.py** — Dynamic video pipeline; real-time overlay + compute-FPS.  
- **part3.py** — Advanced refinement for dynamic video (large-window region consistency + thresholds).  
- **part4.py** — 3D extension: estimates depth from a known circle radius and camera intrinsics, then computes (X, Y, Z).

---

## ▶️ How to Run

Update the input paths inside each script to match your files.  
Look for `IMG_PATH` / `VIDEO_PATH` at the top of each `part*.py`.

---

### 1) Static Image (part1)

Detect shapes from a single image and show contours/centers:

```bash
python3 part1.py
```

Default expects `pennair_static.png` (see `IMG_PATH` in part1.py).

---

### 2) Dynamic Video (Basic) — part2

Process a video stream and show a live overlay with compute FPS:

```bash
python3 part2.py
```

Default expects `pennair_dynamic_ez.mp4` (see `VIDEO_PATH` in part2.py).

---

### 3) Dynamic Video (Advanced Refinement) — part3

Adds large-window region-consistency pruning and optional morphology:

```bash
python3 part3.py
```

Default expects `PennAir 2024 App Dynamic Hard.mp4` (see `VIDEO_PATH`).

Key tunables in `part3.py`:

- `GAUSS_SIGMA` — pre-blur to reduce noise  
- `TOL_X, TOL_Y` — tolerant zero-map thresholds  
- `MIN_ADJ` — neighbor count threshold  
- `PRUNE_N` — small-neighborhood prune radius (1 → 3×3)  
- `LARGE_PRUNE_WIN` — second-stage window size (odd)  
- `FILL_THRESHOLD, CLEAR_THRESHOLD` — hysteresis-like fill/clear  
- `MORPH_CLOSE, MORPH_KERNEL_SIZE` — smoothing  

---

### 4) Simple 3D Localization — part4

Estimates plane depth using a detected reference circle (known real radius), then computes `(X, Y, Z)` of each detected shape:

```bash
python part4.py
```

Default expects `pennair_dynamic_ez.mp4` (edit `VIDEO_PATH`).

Camera intrinsics (in pixels) and real circle radius (meters) are set at top of `part4.py`:

```python
FX = 2564.3186869
FY = 2569.70273111
REAL_RADIUS_M = 0.254  # 10 inches
```

**Assumptions:**

- Pinhole model; principal point at image center of processed frame.  
- Flat scene at average depth estimated over the first ~5 seconds.  
- Reference circle detection used to infer depth:  
  `Z = (FX * REAL_RADIUS_M) / r_px`

---

## My Thought Process

**Initial idea — Masking:**  
At first, masking felt like the simplest approach. But it wouldn’t generalize across scenes.

**Switch to optical flow (forward differences):**  
Given the solid shapes and a background with large pixelwise intensity changes, I used a 1-pixel shift + subtraction (forward difference in x/y). This worked well but left many holes. This should, ideally, work for any solid shape against noisy background, and be extremely computationally efficient.

**Filling the holes:**

- Blur → subtract → invert (0 → 1, nonzero → 0).  
- Convolve with an all-ones kernel (except center).  
- If local white fraction ≥ threshold → flip to white.  
- If local white fraction ≤ threshold → flip to black.  

After contour extraction, discard small blobs (area < 1000 px).

**Dynamic video:**  
Tuned parameters and added a larger prune kernel for robustness. It became less efficient but agnostic to background and shape color.

**3D extension:**  
Leveraged optics: detect a circle, use its known real radius and measured pixel radius to estimate depth (Z), then compute `(X, Y, Z)` via intrinsics.

**ROS2:**  
Began a ROS2 node for publishing detections, but did not complete integration.

---

## Configuration Highlights

Each script exposes key parameters near the top:

**part1/part2/part3**

- `SHIFT` — forward-diff shift (pixels)  
- `GAUSS_SIGMA` — Gaussian blur sigma (0 disables)  
- `TOL_X, TOL_Y` — tolerated |dx|, |dy| thresholds  
- `MIN_ADJ` — neighbor count threshold (> MIN_ADJ keeps pixel)  
- `MIN_AREA` — contour area threshold (px²)  
- `PRUNE_N` — small-neighborhood radius (1 → 3×3, 2 → 5×5, …)  

**part3 (extra)**

- `LARGE_PRUNE_WIN` — large window for regional consistency (odd)  
- `FILL_THRESHOLD, CLEAR_THRESHOLD` — hysteresis flip controls  
- `MORPH_CLOSE, MORPH_KERNEL_SIZE` — optional closing  

**part4 (3D)**

- `FX, FY` — focal lengths in pixels  
- `REAL_RADIUS_M` — real circle radius in meters  
- Uses most circular contour (via circularity `4πA/P²`) for reference  

---

## 🧪 Tips & Troubleshooting

- **If no shapes appear:**  
  - Increase `GAUSS_SIGMA` slightly.  
  - Loosen `TOL_X, TOL_Y`.  
  - Reduce `MIN_AREA` to visualize more candidates.  

- **If too many specks/noise:**  
  - Increase `MIN_ADJ` and/or `PRUNE_N`.  
  - In `part3`, increase `LARGE_PRUNE_WIN` or adjust `FILL/CLEAR_THRESHOLD`.  

- **If 3D depth prints nan or fails:**  
  - Ensure a circle is visible and large enough.  
  - Check `REAL_RADIUS_M` is correct (10 in = 0.254 m).  
  - Verify the video path and that FPS is detected (defaults to 30 if unknown).  



Static Borders:

![Detector overlay](output_videos/static.png)


![Demo](gif_out/OUT_dynamic_ez.gif)



