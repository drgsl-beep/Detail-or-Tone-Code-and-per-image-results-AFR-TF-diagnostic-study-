import sys, numpy, scipy, cv2, skimage
print("Python", sys.version.split()[0])
print("NumPy", numpy.__version__); print("SciPy", scipy.__version__)
print("OpenCV", cv2.__version__); print("scikit-image", skimage.__version__)
try:
    import torch; print("PyTorch", torch.__version__)
except Exception as e: print("PyTorch: not importable", e)
try:
    import lpips; print("lpips", getattr(lpips, "__version__", "check: pip show lpips"))
except Exception as e: print("lpips: not importable", e)
