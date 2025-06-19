import cv2
import argparse
import time
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from collections import defaultdict

def run_live_demo(model_path, video_path, conf_threshold=0.25, save_output=True):
    print(f"Loading model from {model_path}")
    model = YOLO(model_path)
    class_names = model.names
    print(f"Loaded model with {len(class_names)} classes: {class_names}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video properties: {width}x{height} at {fps} FPS, {total_frames} frames")

    if save_output:
        output_path = f"{Path(video_path).stem}_yolo_demo.mp4"
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"Saving output to {output_path}")
    else:
        out = None

    cv2.namedWindow("YOLO Detection Demo", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLO Detection Demo", min(1280, width), min(720, height))

    frame_times = []
    processing_times = []
    total_detections = 0
    frame_count = 0
    print("Starting video processing. Press 'q' to quit, 's' to save a screenshot.")

    while True:
        frame_start = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        process_start = time.time()
        results = model.predict(frame, conf=conf_threshold, verbose=False)[0]
        process_end = time.time()
        processing_time = process_end - process_start
        processing_times.append(processing_time)

        annotated_frame = frame.copy()
        total_detections += len(results.boxes)

        detection_data = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            detection_data.append({"box": box, "cls_id": cls_id, "conf": conf})

        grouped = defaultdict(list)
        for det in detection_data:
            grouped[det["cls_id"]].append(det)

        final_detections = []
        for cls_id, group in grouped.items():
            if len(group) == 2:
                # If two of the same class, demote the one with lower confidence to black (id 2), if it's not already
                group_sorted = sorted(group, key=lambda d: d["conf"], reverse=True)
                final_detections.append(group_sorted[0])
                if group_sorted[1]["cls_id"] != 2:
                    group_sorted[1]["cls_id"] = 2
                    group_sorted[1]["conf"] = min(group_sorted[1]["conf"], 0.49)
                final_detections.append(group_sorted[1])
            else:
                final_detections.extend(group)

        for det in final_detections:
            cls_id = det["cls_id"]
            conf = det["conf"]
            box = det["box"]
            cls_name = class_names.get(cls_id, f"Class {cls_id}")

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            color_factor = cls_id / len(class_names) if len(class_names) > 0 else 0
            color = tuple(int(c) for c in (120 + 120 * np.sin(color_factor * 2 * np.pi), 
                                          120 + 120 * np.sin(color_factor * 2 * np.pi + 2*np.pi/3), 
                                          120 + 120 * np.sin(color_factor * 2 * np.pi + 4*np.pi/3)))

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            label = f"{cls_name}: {conf:.2f}"
            (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(annotated_frame, (x1, y1 - text_height - 10), (x1 + text_width, y1), color, -1)
            cv2.putText(annotated_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        frame_time = time.time() - frame_start
        frame_times.append(frame_time)
        current_fps = 1 / (sum(frame_times[-20:]) / min(len(frame_times), 20)) if frame_times else 0

        info_text = f"Frame: {frame_count}/{total_frames} | FPS: {current_fps:.1f} | Detections: {len(final_detections)}"
        cv2.putText(annotated_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("YOLO Detection Demo", annotated_frame)

        if out is not None:
            out.write(annotated_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Quitting...")
            break
        elif key == ord('s'):
            screenshot_path = f"screenshot_{frame_count:04d}.jpg"
            cv2.imwrite(screenshot_path, annotated_frame)
            print(f"Saved screenshot to {screenshot_path}")

        frame_count += 1

        if frame_count % 100 == 0:
            progress = frame_count / total_frames * 100
            avg_fps = frame_count / sum(frame_times)
            print(f"Progress: {progress:.1f}% | Frame: {frame_count}/{total_frames} | Avg FPS: {avg_fps:.1f}")

    cap.release()
    if out is not None:
        out.release()
    cv2.destroyAllWindows()

    if frame_count > 0:
        avg_process_time = sum(processing_times) / len(processing_times)
        avg_fps = frame_count / sum(frame_times)
        avg_detections = total_detections / frame_count

        print("\n--- Performance Summary ---")
        print(f"Processed {frame_count} frames at {avg_fps:.1f} FPS")
        print(f"Average processing time: {avg_process_time*1000:.1f} ms per frame")
        print(f"Average detections: {avg_detections:.1f} per frame")
        print(f"Total detections: {total_detections}")

        if save_output:
            print(f"Output video saved to: {output_path}")

def parse_args():
    parser = argparse.ArgumentParser(description="Run live YOLO demo with fallback reclassification")
    parser.add_argument("--model", type=str, required=True, help="Path to trained YOLO model (.pt)")
    parser.add_argument("--video", type=str, required=True, help="Path to video file")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--no-save", action="store_true", help="Don't save output video")
    return parser.parse_args()

def main():
    args = parse_args()
    run_live_demo(
        model_path=args.model,
        video_path=args.video,
        conf_threshold=args.conf,
        save_output=not args.no_save
    )

if __name__ == "__main__":
    main()