import cv2
import numpy as np
import matplotlib.pyplot as plt
import shape_detector

# ==== Config ====
IMG_PATH = "pennair_static.png"
SHIFT    = 1        # pixel shift
MIN_ADJ  = 5        # > MIN_ADJ neighbors to keep a pixel (8-neighborhood)
MIN_AREA = 80       # ignore tiny blobs

shape_detector = shape_detector.ShapeDetector()
# ---- Load -> uint8 gray (fast, exact) ----
bgr   = cv2.imread(IMG_PATH, cv2.IMREAD_COLOR)

final_vis = shape_detector.process_bgr(bgr)
plt.imshow(cv2.cvtColor(final_vis, cv2.COLOR_BGR2RGB)); plt.axis("off")
plt.tight_layout(); plt.show()
