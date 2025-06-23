# Import libraries
import time
from exp_video_processing import process_video
import json
import numpy as np

# Evaluate algorithms
def compute_fps(models):
    for model in models:
        start = time.time()
        frame_count = process_video(
            detector=model,
            source_video='vehicles.mp4',
            target_video='vehicles-result.mp4'
        )
        elapsed = time.time() - start
        print(f"{model}: {frame_count/elapsed:.1f} FPS")

def compute_tracking_proxy(log_json_path):
    with open(log_json_path, 'r') as f:
        data = json.load(f)
    tracked = data['tracked_ids']
    switches = data['id_switches']
    total_frames = data['total_frames']

    total_detections = sum(len(frames) for frames in tracked.values())
    dropouts = sum(1 for frames in tracked.values() if len(frames) < total_frames)

    mota_proxy = 1 - (switches + dropouts) / total_detections
    print(f'{log_json_path.split("_")[0]} proxy-MOTA: {mota_proxy:.3f}')

def compute_mean_speed_cv(log_json_path, start=100, end=300):
    with open(log_json_path, 'r') as f:
        data = json.load(f)

    speeds_dict = data['speeds']

    cvs = []
    for tid, v_list in speeds_dict.items():
        arr = np.array(v_list[start:end], dtype=np.float32)
        arr = arr[~np.isnan(arr)]
        if arr.size > 5 and arr.mean() > 0:
            cv = arr.std() / arr.mean()
            cvs.append(cv)

    avg_cv = float(np.mean(cvs)) if cvs else float('nan')
    print(f"{log_json_path.split("_")[0]} average coefficient of variation (cv): {avg_cv}")

# Evaluation
compute_fps(["yolov8x-640", "rfdetr-base", "yolov11x-640"])

compute_tracking_proxy('logs/yolov8x-640_log.json')
compute_tracking_proxy('logs/rfdetr-base_log.json')
compute_tracking_proxy('logs/yolov11x-640_log.json')

compute_mean_speed_cv('logs/yolov8x-640_log.json')
compute_mean_speed_cv('logs/rfdetr-base_log.json')
compute_mean_speed_cv('logs/yolov11x-640_log.json')