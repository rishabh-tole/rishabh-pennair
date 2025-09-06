import cv2
import time
import matplotlib.pyplot as plt
from shape_detector import ShapeDetector

# ========================== Config ==========================
VIDEO_PATH   = "pennair_dynamic_ez.mp4"
SHIFT        = 1       # forward-diff shift
GAUSS_SIGMA  = 1     # Gaussian blur
TOL_X        = 1       # tolerance for |dx|
TOL_Y        = 1       # tolerance for |dy|
MIN_ADJ      = 5       # neighbors threshold (> MIN_ADJ keeps a pixel)
MIN_AREA     = 1000    # minimum area to keep a contour
PRUNE_N      = 1       # neighborhood radius (1 => 3x3)
PAUSE_SEC    = 0.001   # Matplotlib UI update pause

# ===================== Initialize detector ===================
det = ShapeDetector(
    min_area=MIN_AREA,
    adjacent=MIN_ADJ,
    shift=SHIFT,
    prune_neighborhood_n=PRUNE_N
)

# =================== Open the input video ====================
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Could not open {VIDEO_PATH}")

# =================== Prime the display loop ==================
ok, first_frame = cap.read()
if not ok:
    cap.release()
    raise RuntimeError("Empty video or read failed on first frame.")

# Process once to get output size and a first image to show
out0_bgr = det.process_bgr(first_frame, gauss_sigma=GAUSS_SIGMA, tol_x=TOL_X, tol_y=TOL_Y)
# Convert BGR (OpenCV) to RGB (Matplotlib)
out0_rgb = cv2.cvtColor(out0_bgr, cv2.COLOR_BGR2RGB)

# =================== Set up Matplotlib figure =================
plt.ion()  # interactive mode: non-blocking updates
fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(out0_rgb)
ax.set_title("Processed Overlay (Contours + Centers)")
ax.axis("off")
fig.tight_layout()

# =================== Streaming loop with FPS ==================
proc_time_sum = 0.0    # sum of *processing* times (excludes I/O and plotting)
n_proc_frames = 0      # count of frames processed
wall_start = time.perf_counter()  # wall-clock timer for end-to-end throughput

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # ---- Compute-only timer start (pure processing cost) ----
        t0 = time.perf_counter()
        out_bgr = det.process_bgr(frame, gauss_sigma=GAUSS_SIGMA, tol_x=TOL_X, tol_y=TOL_Y)
        proc_time_sum += (time.perf_counter() - t0)
        n_proc_frames += 1

        # ---- Update Matplotlib image buffer (convert BGR->RGB) ----
        out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
        im.set_data(out_rgb)

        # Optional: show a dynamic title with light FPS info
        # (comment out if you want an absolutely static title)
        if proc_time_sum > 0 and n_proc_frames > 0:
            fps_proc = n_proc_frames / proc_time_sum
            ax.set_title(f"Processed Overlay — Compute FPS: {fps_proc:.1f}")

        # ---- Paint the updated frame ----
        plt.pause(PAUSE_SEC)  # allow UI to refresh without blocking

finally:
    # Always release the video and close the figure cleanly
    cap.release()
    plt.ioff()
    plt.close(fig)

# =================== Print simple stats to console ==================
wall_fps = n_proc_frames / (time.perf_counter() - wall_start) if n_proc_frames else 0.0
proc_fps = n_proc_frames / proc_time_sum if proc_time_sum > 0 else 0.0
print(f"Frames processed: {n_proc_frames}")
print(f"Average processing FPS (compute only): {proc_fps:.2f}")
print(f"End-to-end throughput (incl. I/O + plotting): {wall_fps:.2f} FPS")
