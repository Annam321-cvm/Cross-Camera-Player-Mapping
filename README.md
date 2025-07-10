Cross-Camera Player Mapping for Sports Analytics

This repository contains a robust dual-camera player re-identification and tracking system designed for sports analytics, particularly for football (soccer). It processes video feeds from two different camera angles, identifies individual players, and consistently tracks their identities across both views, providing a unified output for analysis.

Features

Dual-Camera Video Processing: Simultaneously processes two video streams (e.g., a broadcast view and a tactical view).
Player Detection: Utilizes a YOLO model (`yolo11n.pt`) to accurately detect players in each frame.
Enhanced Feature Extraction: Extracts unique "fingerprints" for each player, combining color histograms, texture, body part colors, and geometric features for robust re-identification.
Cross-Camera Re-Identification: Employs an advanced `GlobalPlayerTracker` with a Hungarian algorithm-based assignment to link the same player across different camera views, assigning consistent global IDs and colors.
Robust Tracking: Features include a configurable `max_disappear_frames` to handle occlusions and `id_stability_frames` to ensure reliable player ID assignments.
Visual Output: Generates a combined output video where both camera feeds are stacked vertically, displaying detected players with consistent bounding box colors and global IDs across views, along with camera labels and player counts.
Asynchronous Processing: Video analysis runs in a separate thread, allowing for real-time status monitoring in the console.
Configurable Parameters: Easily adjust detection confidence, similarity thresholds, and tracking parameters via a JSON configuration file.
Batch Processing: Supports processing multiple pairs of videos sequentially using a batch configuration file.
Detailed Reporting: Generates comprehensive HTML and JSON reports summarizing system configuration and processing statistics.
Modular Design: Separates concerns into `enhanced_main_script.py` (orchestration, I/O) and `enhanced_runner_script.py` (core CV logic).

Getting Started

Prerequisites

* Python 3.x
* `pip` (Python package installer)
* CUDA-compatible GPU (recommended for YOLO performance)

Installation

1.  opencv-python: Used for computer vision tasks, including reading/writing video frames and drawing annotations.

2.  numpy: Essential for numerical operations, especially array manipulations for image processing and feature vectors.

3.  ultralytics: Specifically, this is used for the YOLO model to perform object detection (player detection).

4.  scipy: Used for scientific computing, specifically for the linear_sum_assignment function (Hungarian algorithm) in the GlobalPlayerTracker for cross-camera player assignment.
 
5.  scikit-learn: Used for machine learning functionalities, particularly cosine_similarity to calculate feature similarities between players.

6.  Download YOLO Model:
    Ensure you have the `yolo11n.pt` model file in your project root directory. This model is used for player detection. You can typically find it on Ultralytics' YOLOv8 GitHub releases or similar sources.

 These dependencies would typically be listed in a requirements.txt file, which has been mentioned as part of my GitHub repository. It can be evetually installed using pip install -r requirements.txt.
 
Project Structure
├── broadcast.mp4                 # input video from broadcast camera
├── tacticam.mp4                  # input video from tactical camera
├── config.yaml                   # configuration file (can be .json too)
├── enhanced_main_script.py       # Main script focussed on orchestration, I/O, and reporting
├── enhanced_runner_script.py     # Core script used for detection, feature extraction, and re-identification
├── global_reid_output.mp4        # output video
├── requirements.txt              # Python dependencies are enlisted within this file
├── yolo11n.pt                    # Pre-trained YOLO model for object detection
└── README.md                     # This actual file you're reading
