from loguru import logger
import cv2
import numpy as np
from typing import Dict, List
from .legacy.theenhancer_colab import EnhancedFingerprintSelector

class EnhancerService:
    def __init__(self):
        self.selector = EnhancedFingerprintSelector()
        logger.info("EnhancerService initialized with methods: {}", list(self.selector.methods.keys()))

    def list_methods(self) -> List[str]:
        return list(self.selector.methods.keys())

    def enhance(self, image_bgr: np.ndarray, method: str) -> np.ndarray:
        if method not in self.selector.methods:
            raise ValueError(f"Unknown method '{method}'. Available: {self.list_methods()}")

        gray = self.selector.preprocess_image(image_bgr)
        enhanced = self.selector.enhance_image(gray, method)
        if enhanced.dtype != np.uint8:
            enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return enhanced

    def enhance_all(self, image_bgr: np.ndarray) -> Dict[str, np.ndarray]:
        gray = self.selector.preprocess_image(image_bgr)
        out = {}
        for m in self.list_methods():
            r = self.selector.enhance_image(gray, m)
            if r.dtype != np.uint8:
                r = cv2.normalize(r, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            out[m] = r
        return out
