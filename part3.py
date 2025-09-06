import cv2
import numpy as np
import time
from shape_detector import ShapeDetector

# ================= CONFIG =================
VIDEO_PATH   = "PennAir 2024 App Dynamic Hard.mp4"

SHIFT        = 1
GAUSS_SIGMA  = 0.5
TOL_X        = 3
TOL_Y        = 3
MIN_ADJ      = 17
MIN_AREA     = 2000
PRUNE_N      = 2
LARGE_PRUNE_WIN    = 45      # odd kernel size
FILL_THRESHOLD     = 0.7     # if black & local white-fraction >= this -> flip to white
CLEAR_THRESHOLD    = 0.33    # if white & local white-fraction <= this -> flip to black
MORPH_CLOSE        = False    # optionally smooth shapes with a closing after refine
MORPH_KERNEL_SIZE  = 3       # closing kernel (odd)

# ==========================================

det = ShapeDetector(min_area=MIN_AREA, adjacent=MIN_ADJ, shift=SHIFT, prune_neighborhood_n=PRUNE_N)

def refine_mask_region_consistency(mask255: np.ndarray,
                                   win: int,
                                   fill_thresh: float,
                                   clear_thresh: float,
                                   do_close: bool = False,
                                   close_ksize: int = 3) -> np.ndarray:
    """
    Second-stage prune: density in a larger window.
      - mask255: uint8, 0/255
      - win: odd window size for local counting
      - fill_thresh: if pixel is 0 and local white-fraction >= this, set to 255
      - clear_thresh: if pixel is 255 and local white-fraction <= this, set to 0
      - do_close: optional morphological closing to smooth edges
    """
    win = int(win) if win % 2 == 1 else int(win) + 1
    k = np.ones((win, win), dtype=np.uint8)

    # Work in 0/1 domain for counting
    m01 = (mask255 > 0).astype(np.uint8)

    # Local white count per window (includes center)
    local_white = cv2.filter2D(m01, ddepth=cv2.CV_16U, kernel=k, borderType=cv2.BORDER_CONSTANT)
    area = win * win
    frac = local_white.astype(np.float32) / float(area)

    # Hysteresis update
    fill = (m01 == 0) & (frac >= fill_thresh)
    clear = (m01 == 1) & (frac <= clear_thresh)
    keep = ~(fill | clear)

    refined01 = np.zeros_like(m01, dtype=np.uint8)
    refined01[fill] = 1
    refined01[keep] = m01[keep]
    # clear already zero

    refined255 = (refined01 * 255).astype(np.uint8)

    if do_close and close_ksize and close_ksize >= 3:
        if close_ksize % 2 == 0:
            close_ksize += 1
        ck = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ksize, close_ksize))
        refined255 = cv2.morphologyEx(refined255, cv2.MORPH_CLOSE, ck)

    return refined255

# open video
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"Could not open {VIDEO_PATH}")

# create resizable OpenCV windows
cv2.namedWindow("Mask (refined)", cv2.WINDOW_NORMAL)
cv2.namedWindow("final neighbor prune", cv2.WINDOW_NORMAL)  # ADDED
cv2.namedWindow("Overlay", cv2.WINDOW_NORMAL)

# FPS measurement (EMA smoothing)
fps = 0.0
ema_alpha = 0.2  # higher = snappier updates
prev_time = time.time()

font = cv2.FONT_HERSHEY_SIMPLEX
txt_scale = 0.6
txt_th = 2
txt_color = (255, 255, 255)
txt_shadow = (0, 0, 0)

try:
    while True:
        loop_start = time.time()

        ok, frame = cap.read()
        if not ok:
            break

        # --- pipeline (first prune from class) ---
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur  = cv2.GaussianBlur(gray, (0, 0), GAUSS_SIGMA, GAUSS_SIGMA,
                                 borderType=cv2.BORDER_REPLICATE) if GAUSS_SIGMA > 0 else gray
        
    

        dx, dy = det.forward_diff(blur)
        h, w   = min(dx.shape[0], dy.shape[0]), min(dx.shape[1], dy.shape[1])
        bgr_c  = frame[:h, :w]
        M      = det.zero_map(dx, dy, TOL_X, TOL_Y)
        mask   = det.neighbor_prune(M)              # uint8 0/255



        # --- NEW: second-stage region-consistency prune ---
        mask_refined = refine_mask_region_consistency(
            mask,
            win=LARGE_PRUNE_WIN,
            fill_thresh=FILL_THRESHOLD,
            clear_thresh=CLEAR_THRESHOLD,
            do_close=MORPH_CLOSE,
            close_ksize=MORPH_KERNEL_SIZE
        )


        # Draw contours using refined mask
        overlay = det.draw_contours(bgr_c, mask_refined)

        # --- FPS calculation ---
        now = time.time()
        inst_fps = 1.0 / max(now - loop_start, 1e-6)
        fps = ema_alpha * inst_fps + (1.0 - ema_alpha) * fps

        # Put FPS text onto all displayed images
        def draw_fps(img, label):
            txt = f"{label} | {fps:5.1f} FPS"
            # shadow
            cv2.putText(img, txt, (10, 24), font, txt_scale, txt_shadow, txt_th+2, cv2.LINE_AA)
            # main
            cv2.putText(img, txt, (10, 24), font, txt_scale, txt_color, txt_th, cv2.LINE_AA)

        # Prepare display copies
        mask_ref_disp = cv2.cvtColor(mask_refined, cv2.COLOR_GRAY2BGR)
        overlay_disp  = overlay.copy()

        draw_fps(mask_ref_disp, "Mask (refined)")
        draw_fps(overlay_disp,  "Overlay")

        # Also update window titles with FPS (works on most builds)
        cv2.setWindowTitle("Mask (refined)",       f"Mask (refined) — {fps:0.1f} FPS")
        cv2.setWindowTitle("Overlay",              f"Overlay — {fps:0.1f} FPS")

        # Show frames
        cv2.imshow("Overlay", overlay_disp)

        # exit on 'q' or ESC
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q')):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
