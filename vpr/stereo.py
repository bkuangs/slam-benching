"""Stereo depth for database frames: OpenCV SGBM on the rectified pair with a left-right consistency check."""
import cv2
import numpy as np


class StereoDepth:
    def __init__(self, fx: float, baseline_m: float, num_disp: int = 64, block: int = 5, lr_tol_px: float = 1.0):
        self.fb = fx * baseline_m
        self.lr_tol = lr_tol_px
        P1, P2 = 8 * block * block, 32 * block * block
        self.sgbm = cv2.StereoSGBM_create(minDisparity=0, numDisparities=num_disp, blockSize=block, P1=P1, P2=P2,
                                          disp12MaxDiff=-1, uniquenessRatio=10, speckleWindowSize=100,
                                          speckleRange=2, mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

    def disparity(self, left_gray, right_gray):
        dl = self.sgbm.compute(left_gray, right_gray).astype(np.float32) / 16.0
        dr = self.sgbm.compute(right_gray[:, ::-1].copy(), left_gray[:, ::-1].copy())[:, ::-1].astype(np.float32) / 16
        h, w = dl.shape
        xs = np.arange(w)[None].repeat(h, 0)
        xr = np.clip(np.round(xs - dl).astype(int), 0, w - 1)
        dr_at = np.take_along_axis(dr, xr, axis=1)
        ok = (dl > 0) & (dr_at > 0) & (np.abs(dl - dr_at) <= self.lr_tol)
        return np.where(ok, dl, 0.0)

    def depth_mm(self, left_bgr, right_bgr, min_disp: float = 0.5) -> np.ndarray:
        """float32 z-depth in mm (0 = invalid), same convention as the sim depth0 PNGs."""
        d = self.disparity(cv2.cvtColor(left_bgr, cv2.COLOR_BGR2GRAY), cv2.cvtColor(right_bgr, cv2.COLOR_BGR2GRAY))
        z = np.zeros_like(d)
        ok = d > min_disp
        z[ok] = 1000.0 * self.fb / d[ok]
        return z
