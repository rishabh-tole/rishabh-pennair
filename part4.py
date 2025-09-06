import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

# ========================== Config ==========================
VIDEO_PATH = "pennair_dynamic_ez.mp4"

# simple knobs
SHIFT, GAUSS_SIGMA = 1, 1
TOL_X, TOL_Y = 1, 1
MIN_ADJ, MIN_AREA, PRUNE_N = 5, 1000, 1
PAUSE_SEC = 0.001

# camera intrinsics (pixels) + known real radius of reference circle (meters)
FX = 2564.3186869
FY = 2569.70273111
REAL_RADIUS_M = 0.254  # 10 inches

# ========================== Minimal helpers ==========================
def forward_diff(gray, s=SHIFT):
    dx = gray[:, s:].astype(np.int16) - gray[:, :-s].astype(np.int16)
    dy = gray[s:, :].astype(np.int16) - gray[:-s, :].astype(np.int16)
    return dx, dy

def zero_map(dx, dy, tol_x=TOL_X, tol_y=TOL_Y):
    h = min(dx.shape[0], dy.shape[0])
    w = min(dx.shape[1], dy.shape[1])
    M = ((np.abs(dx[:h, :w]) <= tol_x) & (np.abs(dy[:h, :w]) <= tol_y)).astype(np.uint8)
    return M, h, w

def neighbor_prune(M, prune=PRUNE_N, min_adj=MIN_ADJ):
    size = prune * 2 + 1
    k = np.ones((size, size), dtype=np.uint8); k[prune, prune] = 0
    nbr = cv2.filter2D(M, cv2.CV_16U, k, borderType=cv2.BORDER_CONSTANT)
    return ((M > 0) & (nbr > min_adj)).astype(np.uint8) * 255

def center_from_contour(c):
    m = cv2.moments(c)
    if m["m00"] > 0:
        return (m["m10"]/m["m00"], m["m01"]/m["m00"])
    x, y, w, h = cv2.boundingRect(c)
    return (x + w/2.0, y + h/2.0)

def circularity(c):
    A = cv2.contourArea(c); P = cv2.arcLength(c, True)
    if A <= 0 or P <= 1e-6: return 0.0
    # C = 4πA / P²  (circle≈1, elongated≪1)
    return 4.0 * math.pi * A / (P * P)

def pixel_radius(c):
    (_, _), r = cv2.minEnclosingCircle(c)
    return float(r)

def pick_most_circular(cnts):
    best = None  # (C, u, v, r_px, contour)
    for c in cnts:
        if cv2.contourArea(c) < MIN_AREA: continue
        C = circularity(c); r_px = pixel_radius(c); u, v = center_from_contour(c)
        if best is None or C > best[0]: best = (C, u, v, r_px, c)
    if best is None or best[0] < 0.5 or best[3] <= 1.0: return None
    return best  # (C, u, v, r_px, contour)

def detect_contours_and_circle(frame):
    """
    Returns (h, w, cnts, picked_circle) where:
      h,w: processed frame size (after diff overlap)
      cnts: list of surviving contours
      picked_circle: (C, u, v, r_px, contour) for most-circular or None
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if GAUSS_SIGMA > 0:
        gray = cv2.GaussianBlur(gray, (0, 0), GAUSS_SIGMA, GAUSS_SIGMA, borderType=cv2.BORDER_REPLICATE)
    dx, dy = forward_diff(gray)
    M, h, w = zero_map(dx, dy)
    mask = neighbor_prune(M)
    info = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = info[0] if len(info) == 2 else info[1]
    picked = pick_most_circular(cnts)
    return h, w, cnts, picked

# ========================== First pass: average Z over 5s ==========================
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
if not fps or fps <= 1e-3 or np.isnan(fps): fps = 30.0
init_frames = int(round(5.0 * fps))

Z_samples = []
overlay0 = None
for _ in range(init_frames):
    ok, frame = cap.read()
    overlay0 = frame
    h, w, cnts, picked = detect_contours_and_circle(frame)
    if picked is not None:
        _, u, v, r_px, _ = picked
        # Depth from circle radius
        # r_px / FX = R / Z  =>  Z = (FX * R) / r_px
        Z = (FX * REAL_RADIUS_M) / float(r_px)
        if np.isfinite(Z): Z_samples.append(Z)

if len(Z_samples) == 0:
    cap.release()
    raise RuntimeError("Could not estimate depth in the first 5 seconds (no suitable circle).")

plane_Z = float(np.mean(Z_samples))  # frozen plane depth in meters

# ========================== Display loop ==========================
plt.ion()
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis("off")
im = ax.imshow(cv2.cvtColor(overlay0, cv2.COLOR_BGR2RGB))
ax.set_title(f"Depth (avg first 5s): {plane_Z:.3f} m")
fig.tight_layout()

while True:
    ok, frame = cap.read()
    if not ok: break

    h, w, cnts, picked = detect_contours_and_circle(frame)
    overlay = frame.copy()
    cx, cy = w/2.0, h/2.0  # principal point (processed frame center)

    # Label bottom-left image origin
    cv2.putText(overlay, "(0,0) at bottom-left", (6, h-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)

    # Mark image center and show frozen depth there (optional)
    cv2.drawMarker(overlay, (int(round(cx)), int(round(cy))), (255,255,0),
                   markerType=cv2.MARKER_CROSS, markerSize=16, thickness=2)
    cv2.putText(overlay, f"Z={plane_Z:.3f} m", (int(cx)+10, int(cy)-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2, cv2.LINE_AA)

    # For each contour: draw outline, mark center, and label (X,Y,Z) in meters
    for c in cnts:
        if cv2.contourArea(c) < MIN_AREA: continue

        # Outline
        cv2.drawContours(overlay, [c], -1, (255,255,255), 2)

        # Center in pixels (top-left origin)
        u, v = center_from_contour(c)
        cv2.circle(overlay, (int(round(u)), int(round(v))), 4, (0,0,255), -1)

        # --- Camera-frame translation (meters), X right, Y up, Z forward ---
        # X = ((u - cx)/FX) * Z
        # Y = ((cy - v)/FY) * Z    (because image v grows downward)
        X = ((u - cx) / FX) * plane_Z
        Y = ((cy - v) / FY) * plane_Z
        Z = plane_Z

        # Label (X,Y,Z) near the center
        cv2.putText(overlay, f"({X:.3f}, {Y:.3f}, {Z:.3f}) m",
                    (int(u)+6, int(v)-6), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0,255,0), 1, cv2.LINE_AA)

    im.set_data(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    plt.pause(PAUSE_SEC)

cap.release()
plt.ioff()
plt.close(fig)
