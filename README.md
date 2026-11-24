# VisionTrack AI

**Intelligent Video Object Detection, Tracking & Analytics System**

VisionTrack AI uses **YOLO11n** for object detection and **ByteTrack** for multi-object tracking. It processes an uploaded video, draws detection annotations, records frame-wise object counts, and generates CSV reports and visual analytics.

## Features

- Object detection using the pretrained YOLO11n model
- Multi-object tracking using ByteTrack
- Annotated output video with object labels and a VisionTrack AI overlay
- Frame-wise detection analytics exported to CSV
- Unique tracked-ID summary grouped by object class
- Matplotlib charts for object counts and detection trends
- Google Colab upload and download workflow

## Technology stack

- Python
- Ultralytics YOLO11
- ByteTrack
- OpenCV
- Pandas
- Matplotlib
- Google Colab / IPython

## Run the project

This project is designed for **Google Colab** because it uses `google.colab.files` to upload the input video and download the generated reports.

1. Upload `VisionTrack_AI.ipynb` to Google Colab, or open it from your GitHub repository.
2. Select a Python runtime. A GPU runtime may improve processing speed but is not mandatory.
3. Run the notebook cell.
4. When prompted, upload a short video (MP4 is recommended).
5. Wait for detection, tracking, analytics, and chart generation to finish.
6. Download the processed video and CSV reports when prompted.

The notebook installs its required Python packages at runtime. If you are running the script in another environment, install dependencies with:

```bash
pip install -r requirements.txt
```

**Note:** `visiontrack_ai.py` is provided as a script copy for reference. The included workflow expects Google Colab/IPython and is not a general-purpose command-line application without adapting the upload, display, and download sections.

## Generated outputs

The notebook writes files to `/content/visiontrack_results/` in the Colab runtime:

| File | Description |
|---|---|
| `visiontrack_processed.mp4` | Video with object detection annotations |
| `frame_analytics.csv` | Frame number, timestamp, total detections, and per-class visible counts |
| `object_summary.csv` | Unique observed tracking IDs grouped by object class |

## Important interpretation notes

- The unique-object summary counts distinct tracking IDs returned by ByteTrack during the video. Tracking IDs can occasionally change or be reassigned, so this is an estimate of unique tracked instances rather than a guaranteed count of real-world individuals.
- Frame-wise counts represent detections visible in each processed frame. They are not unique-object counts.
- Detection quality depends on video quality, lighting, camera angle, object size, and the model's confidence threshold.
- The pretrained model weights (`yolo11n.pt`) are downloaded by Ultralytics when the model is first loaded; they are not included in this repository.
- The code has been packaged from the supplied source. It has not been executed or validated against a sample video as part of packaging.

## Repository structure

```text
VisionTrack-AI/
├── VisionTrack_AI.ipynb
├── visiontrack_ai.py
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

## License

This repository includes the MIT License. See [LICENSE](LICENSE).

## Acknowledgements

- Ultralytics for YOLO and its tracking integration
- ByteTrack for multi-object tracking
- OpenCV, Pandas, and Matplotlib for video processing and analytics

Check the licenses and terms of the third-party libraries and pretrained model before redistributing them or using them commercially.
