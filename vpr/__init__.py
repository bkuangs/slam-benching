"""Visual place recognition (VPR) benchmark: learned global descriptors vs DBoW2 bag-of-words retrieval.

Pure-numpy parts (importable without torch): metrics, dbow, sessions.
Torch parts (run inside the LAC image): models, stereo.
See docs/results/place_recognition.md.
"""
