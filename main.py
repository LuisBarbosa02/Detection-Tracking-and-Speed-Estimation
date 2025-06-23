# Import libraries
import argparse
from inference.models.utils import get_roboflow_model
import supervision as sv
import cv2
import numpy as np
from collections import defaultdict, deque

# Config
SOURCE = np.array([[1252, 787], [2298, 803], [5039, 2159], [-550, 2159]])

TARGET_WIDTH = 25  # meters (real-world width)
TARGET_HEIGHT = 250  # meters (real-world height)

TARGET = np.array(
    [
        [0, 0],
        [TARGET_WIDTH - 1, 0],
        [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
        [0, TARGET_HEIGHT - 1]
    ]
)

SPEED_LIMIT_KMH = 120  # km/h

# Helper algorithms
class ViewTransformer:
    def __init__(self, source, target):
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points):
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        if transformed_points is None:
            return np.zeros((0, 2), dtype=np.float32)
        return transformed_points.reshape(-1, 2)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_video_path",
        required=True,
        help="Path to source video file.",
        type=str
    )
    parser.add_argument(
        "--target_video_path",
        required=True,
        help="Path to output annotated video file.",
        type=str
    )
    parser.add_argument(
        "--model",
        required=False,
        default="yolov11x-640",
        help="Detection model to be used.",
        type=str
    )
    return parser.parse_args()

# Run algorithm
if __name__ == '__main__':
    args = parse_arguments()

    video_info = sv.VideoInfo.from_video_path(args.source_video_path)
    if args.model not in ["yolov8x-640", "rfdetr-base", "yolov11x-640"]:
        raise ValueError("Choose between the models: yolov8x-640, rfdetr-base, or yolov11x-640")
    model = get_roboflow_model(args.model)

    byte_track = sv.ByteTrack(frame_rate=video_info.fps)

    thickness = sv.calculate_optimal_line_thickness(resolution_wh=video_info.resolution_wh)
    text_scale = sv.calculate_optimal_text_scale(resolution_wh=video_info.resolution_wh)

    # Green annotators for non-speeding
    bbox_annotator_green = sv.BoundingBoxAnnotator(thickness=thickness, color=sv.Color.GREEN)
    label_annotator_green = sv.LabelAnnotator(
        text_scale=text_scale,
        text_thickness=thickness,
        text_position=sv.Position.BOTTOM_CENTER,
        color=sv.Color.GREEN
    )
    trace_annotator_green = sv.TraceAnnotator(
        thickness=thickness,
        trace_length=video_info.fps * 2,
        position=sv.Position.BOTTOM_CENTER,
        color=sv.Color.GREEN
    )

    # Red annotators for speeding
    bbox_annotator_red = sv.BoundingBoxAnnotator(thickness=thickness, color=sv.Color.RED)
    label_annotator_red = sv.LabelAnnotator(
        text_scale=text_scale,
        text_thickness=thickness,
        text_position=sv.Position.BOTTOM_CENTER,
        color=sv.Color.RED
    )
    trace_annotator_red = sv.TraceAnnotator(
        thickness=thickness,
        trace_length=video_info.fps * 2,
        position=sv.Position.BOTTOM_CENTER,
        color=sv.Color.RED
    )

    polygon_zone = sv.PolygonZone(SOURCE)
    view_transformer = ViewTransformer(source=SOURCE, target=TARGET)

    coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))

    frame_generator = sv.get_video_frames_generator(args.source_video_path)
    with sv.VideoSink(args.target_video_path, video_info) as sink:
        for frame in frame_generator:
            # Detection and tracking
            result = model.infer(frame)[0]
            detections = sv.Detections.from_inference(result)
            detections = detections[polygon_zone.trigger(detections)]
            detections = byte_track.update_with_detections(detections=detections)

            points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
            points = view_transformer.transform_points(points=points).astype(int)

            labels = []
            is_speeding = {}
            for tracker_id, [_, y] in zip(detections.tracker_id, points):
                coords = coordinates[tracker_id]
                coords.append(y)
                if len(coords) < video_info.fps / 2:
                    labels.append(f"#{tracker_id}")
                    is_speeding[tracker_id] = False
                else:
                    distance_m = abs(coords[-1] - coords[0])
                    time_s = len(coords) / video_info.fps
                    speed_kmh = distance_m / time_s * 3.6
                    labels.append(f"#{tracker_id} {int(speed_kmh)} km/h")
                    is_speeding[tracker_id] = speed_kmh > SPEED_LIMIT_KMH

            annotated_frame = frame.copy()

            # Split indices
            non_speed_idx, speed_idx = [], []
            for i, tid in enumerate(detections.tracker_id):
                (speed_idx if is_speeding.get(tid, False) else non_speed_idx).append(i)

            # Non-speeding annotations
            if non_speed_idx:
                det_non = detections[non_speed_idx]
                lbl_non = [labels[i] for i in non_speed_idx]
                annotated_frame = trace_annotator_green.annotate(scene=annotated_frame, detections=det_non)
                annotated_frame = bbox_annotator_green.annotate(scene=annotated_frame, detections=det_non)
                annotated_frame = label_annotator_green.annotate(scene=annotated_frame, detections=det_non, labels=lbl_non)

            # Speeding annotations
            if speed_idx:
                det_spd = detections[speed_idx]
                lbl_spd = [labels[i] for i in speed_idx]
                annotated_frame = trace_annotator_red.annotate(scene=annotated_frame, detections=det_spd)
                annotated_frame = bbox_annotator_red.annotate(scene=annotated_frame, detections=det_spd)
                annotated_frame = label_annotator_red.annotate(scene=annotated_frame, detections=det_spd, labels=lbl_spd)

            sink.write_frame(annotated_frame)
            cv2.imshow("Frame", annotated_frame)
            if cv2.waitKey(1) == ord('q'):
                break
    cv2.destroyAllWindows()