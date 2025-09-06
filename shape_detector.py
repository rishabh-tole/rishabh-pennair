import cv2
import numpy as np

class ShapeDetector:
    def __init__(self, min_area=1000, adjacent=5, shift=1, prune_neighborhood_n=1):
        # Keep only contours with area >= min_area
        self.min_area = int(min_area)
        # For pruning: pixel survives if it has > adjacent neighbors ON
        self.min_adjacent = int(adjacent)
        # Forward-difference shift (# of pixels)
        self.shift = int(shift)
        # Neighborhood radius: 1 => 3x3, 2 => 5x5, ...
        self.prune = int(prune_neighborhood_n)

    def forward_diff(self, gray):
        """Compute forward finite differences along x and y (int16 to avoid uint8 wrap)."""
        s = self.shift
        dx = gray[:, s:].astype(np.int16) - gray[:, :-s].astype(np.int16)
        dy = gray[s:, :].astype(np.int16) - gray[:-s, :].astype(np.int16)
        return dx, dy

    def zero_map(self, dx, dy, tol_x=1, tol_y=1):
        """Binary map (0/1): 1 where |dx|<=tol_x AND |dy|<=tol_y over common overlap."""
        h = min(dx.shape[0], dy.shape[0])
        w = min(dx.shape[1], dy.shape[1])
        M = ((np.abs(dx[:h, :w]) <= tol_x) & (np.abs(dy[:h, :w]) <= tol_y)).astype(np.uint8)
        return M

    def neighbor_prune(self, M):
        """Count neighbors (excluding center) then keep pixels with count > min_adjacent."""
        size = self.prune * 2 + 1               # e.g., 3 for prune=1
        k = np.ones((size, size), dtype=np.uint8)
        k[self.prune, self.prune] = 0           # don't count the center pixel
        nbr = cv2.filter2D(M, cv2.CV_16U, k, borderType=cv2.BORDER_CONSTANT)
        mask = ((M > 0) & (nbr > self.min_adjacent)).astype(np.uint8) * 255
        return mask

    def draw_contours(self, bgr_c, mask):
        """Draw surviving contours and their centers on a copy of the input color image."""
        overlay = bgr_c.copy()
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            if cv2.contourArea(c) < self.min_area:
                continue
            cv2.drawContours(overlay, [c], -1, (255, 255, 255), 2)
            m = cv2.moments(c)
            if m["m00"] > 0:
                cx, cy = int(m["m10"]/m["m00"]), int(m["m01"]/m["m00"])
            else:
                x, y, w0, h0 = cv2.boundingRect(c)
                cx, cy = x + w0//2, y + h0//2
            cv2.circle(overlay, (cx, cy), 4, (0, 0, 255), -1)
        return overlay

    def process_bgr(self, bgr, gauss_sigma=0.0, tol_x=1, tol_y=1):
        """
        Full pipeline for a single frame:
        1) Gray (uint8) 2) Optional Gaussian blur 3) Forward diffs 4) Zero-map 5) Prune
        6) Draw contours/centers on cropped color image
        Returns: overlay (BGR) cropped to the valid diff region.
        """
        # 1) Convert to grayscale
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

        # 2) Optional Gaussian blur to reduce noise (sigma=0 -> no blur)
        blur = (cv2.GaussianBlur(gray, (0, 0), gauss_sigma, gauss_sigma,
                                 borderType=cv2.BORDER_REPLICATE)
                if gauss_sigma > 0 else gray)

        # 3) Forward finite differences
        dx, dy = self.forward_diff(blur)

        # 4) Determine overlap region (diffs shrink image by 'shift' along each axis)
        h = min(dx.shape[0], dy.shape[0])
        w = min(dx.shape[1], dy.shape[1])
        bgr_c = bgr[:h, :w]  # crop original color image to match diff overlap

        # 5) Tolerant zero-map + 6) Neighbor prune to remove specks
        M = self.zero_map(dx, dy, tol_x, tol_y)  # uint8 0/1
        mask = self.neighbor_prune(M)            # uint8 0/255

        # 7) Draw contours and centers on color image
        overlay = self.draw_contours(bgr_c, mask)

        return overlay
