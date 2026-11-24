# ================================================================
# VISIONTRACK AI
# Intelligent Video Object Detection, Tracking & Analytics System
# YOLO11n + ByteTrack + OpenCV + Pandas + Matplotlib
# ================================================================

import os
import sys
import subprocess
import time
from pathlib import Path
from collections import Counter

print("=" * 65)
print("             VISIONTRACK AI")
print(" Intelligent Video Detection, Tracking & Analytics")
print("=" * 65)

# ------------------------------------------------
# 1. INSTALL REQUIRED LIBRARIES
# ------------------------------------------------
print("\n[1/7] Installing required libraries...")

subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "ultralytics", "opencv-python-headless", "pandas", "matplotlib"
])

import cv2
import pandas as pd
import matplotlib.pyplot as plt

from ultralytics import YOLO
from google.colab import files
from IPython.display import display, Video

print("Libraries installed successfully!")

# ------------------------------------------------
# 2. LOAD PRETRAINED AI MODEL
# ------------------------------------------------
print("\n[2/7] Loading YOLO11n AI model...")

model = YOLO("yolo11n.pt")

print("AI model loaded successfully!")
print("Detection model: YOLO11n")
print("Tracking algorithm: ByteTrack")

# ------------------------------------------------
# 3. UPLOAD INPUT VIDEO
# ------------------------------------------------
print("\n[3/7] Upload your video when prompted.")
print("Recommended: A short MP4 video containing people or vehicles.")

uploaded = files.upload()

if not uploaded:
    raise ValueError("No video uploaded. Run the cell again and select a video.")

video_path = next(iter(uploaded.keys()))

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise ValueError(
        "Unable to open the uploaded video. Please try an MP4 video."
    )

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if fps <= 0:
    fps = 25.0

if width <= 0 or height <= 0:
    cap.release()
    raise ValueError("Invalid video dimensions.")

duration = total_frames / fps if total_frames > 0 else 0

print("\nInput video information")
print("-" * 40)
print("Filename:", video_path)
print("Resolution:", f"{width} x {height}")
print("Frame rate:", round(fps, 2), "FPS")
print("Total frames:", total_frames)
print("Duration:", round(duration, 2), "seconds")

# ------------------------------------------------
# 4. PROCESS VIDEO: DETECTION + OBJECT TRACKING
# ------------------------------------------------
print("\n[4/7] Starting detection and tracking...")
print("Please wait while the video is processed.\n")

output_dir = Path("/content/visiontrack_results")
output_dir.mkdir(parents=True, exist_ok=True)

output_video = str(output_dir / "visiontrack_processed.mp4")
frame_csv = str(output_dir / "frame_analytics.csv")
summary_csv = str(output_dir / "object_summary.csv")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(
    output_video,
    fourcc,
    fps,
    (width, height)
)

if not writer.isOpened():
    cap.release()
    raise RuntimeError(
        "Could not initialize the output video writer. "
        "Try uploading a standard MP4 video."
    )

# Tracking and analytics storage
unique_objects = {}
frame_records = []
frame_number = 0
start_time = time.time()

try:
    while True:
        success, frame = cap.read()

        if not success:
            break

        frame_number += 1

        # Run YOLO object detection and ByteTrack tracking
        results = model.track(
            source=frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=0.35,
            verbose=False
        )

        result = results[0]

        # Generate annotated frame with bounding boxes and labels
        annotated = result.plot()

        # Count objects visible in this frame
        visible_counts = Counter()
        boxes = result.boxes

        if boxes is not None and len(boxes) > 0:
            class_ids = boxes.cls.int().cpu().tolist()

            if boxes.id is not None:
                track_ids = boxes.id.int().cpu().tolist()

                for track_id, class_id in zip(track_ids, class_ids):
                    class_name = model.names[int(class_id)]

                    # Maintain a record of unique track IDs
                    unique_objects[int(track_id)] = class_name

                    visible_counts[class_name] += 1

            else:
                # Count detections even when tracking IDs are unavailable
                for class_id in class_ids:
                    class_name = model.names[int(class_id)]
                    visible_counts[class_name] += 1

        # Overlay project name
        cv2.rectangle(annotated, (10, 8), (340, 53), (30, 30, 30), -1)

        cv2.putText(
            annotated,
            "VISIONTRACK AI",
            (20, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2
        )

        # Add processing information
        cv2.putText(
            annotated,
            f"Frame: {frame_number}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # Write processed frame to output video
        writer.write(annotated)

        # Save per-frame analytics
        record = {
            "Frame": frame_number,
            "Time_seconds": round((frame_number - 1) / fps, 3),
            "Visible_detections": sum(visible_counts.values())
        }

        for class_name, count in visible_counts.items():
            record[class_name] = count

        frame_records.append(record)

        if frame_number % 25 == 0:
            elapsed = time.time() - start_time
            print(
                f"Processed {frame_number} frames | "
                f"Unique tracked IDs: {len(unique_objects)} | "
                f"Elapsed: {elapsed:.1f}s"
            )

finally:
    cap.release()
    writer.release()

processing_time = time.time() - start_time

if frame_number == 0 or not os.path.exists(output_video):
    raise RuntimeError("No video frames were processed.")

print("\nVideo processing completed!")
print("Frames processed:", frame_number)
print("Processing time:", round(processing_time, 2), "seconds")

# ------------------------------------------------
# 5. GENERATE REPORTS AND OBJECT STATISTICS
# ------------------------------------------------
print("\n[5/7] Generating analytics reports...")

frame_df = pd.DataFrame(frame_records).fillna(0)

# Convert count columns to integers
for column in frame_df.columns:
    if column not in ["Frame", "Time_seconds"]:
        frame_df[column] = frame_df[column].astype(int)

frame_df.to_csv(frame_csv, index=False)

class_counts = Counter(unique_objects.values())

summary_df = pd.DataFrame(
    [
        {
            "Object_Class": class_name,
            "Unique_Tracked_Objects": count
        }
        for class_name, count in sorted(class_counts.items())
    ],
    columns=["Object_Class", "Unique_Tracked_Objects"]
)

summary_df.to_csv(summary_csv, index=False)

# ------------------------------------------------
# 6. DISPLAY DASHBOARD AND ANALYTICS CHARTS
# ------------------------------------------------
print("\n[6/7] Creating analytics dashboard...")

print("\n" + "=" * 65)
print("             VISIONTRACK AI REPORT")
print("=" * 65)

print("Input video:", video_path)
print("Frames processed:", frame_number)
print("Video duration:", round(frame_number / fps, 2), "seconds")
print("Unique tracked IDs:", len(unique_objects))
print("Detected object classes:", len(class_counts))

print("\nUnique tracked objects by class:")

if not summary_df.empty:
    display(summary_df)
else:
    print("No objects with tracking IDs were recorded.")

print("\nFrame-wise analytics preview:")
display(frame_df.head(10))

# Chart 1: Unique objects by class
if not summary_df.empty:
    plt.figure(figsize=(11, 5))

    plt.bar(
        summary_df["Object_Class"],
        summary_df["Unique_Tracked_Objects"]
    )

    plt.title("VisionTrack AI - Unique Tracked Objects")
    plt.xlabel("Object Class")
    plt.ylabel("Number of Unique Track IDs")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
else:
    print("Object-count chart unavailable because no track IDs were recorded.")

# Chart 2: Visible detections over time
plt.figure(figsize=(12, 5))

plt.plot(
    frame_df["Time_seconds"],
    frame_df["Visible_detections"]
)

plt.title("Object Detections Across Video Frames")
plt.xlabel("Time (seconds)")
plt.ylabel("Number of Visible Detections")
plt.grid(True)
plt.tight_layout()
plt.show()

# Chart 3: Class-wise detections over time
class_columns = [
    column for column in frame_df.columns
    if column not in [
        "Frame", "Time_seconds", "Visible_detections"
    ]
]

if class_columns:
    plt.figure(figsize=(12, 6))

    for class_name in class_columns:
        plt.plot(
            frame_df["Time_seconds"],
            frame_df[class_name],
            label=class_name
        )

    plt.title("Object Class Detection Trends")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Visible Objects Per Frame")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ------------------------------------------------
# 7. DISPLAY VIDEO AND DOWNLOAD OUTPUT FILES
# ------------------------------------------------
print("\n[7/7] Displaying processed video...")

print("\n" + "=" * 65)
print("VISIONTRACK AI - SUCCESS")
print("=" * 65)
print("Processed video:", output_video)
print("Frame analytics:", frame_csv)
print("Object summary:", summary_csv)
print("=" * 65)

# Display the annotated output video
display(Video(output_video, embed=True, width=800))

# Download the three generated files
print("\nDownloading processed video and reports...")

files.download(output_video)
files.download(frame_csv)
files.download(summary_csv)

print("\nALL TASKS COMPLETED SUCCESSFULLY!")
print("VisionTrack AI has finished processing your video.")
