from ultralytics import YOLO
import cv2
import json

class ObjectDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracked_ids = set()
        self.zone_counts = {}
        self.zones = self.load_zones()
        self.logs = []

    def load_zones(self):
        with open("zones.json") as f:
            return json.load(f)

    def inside_zone(self, x, y, zone):
        x1, y1, x2, y2 = self.zones[zone]
        return x1 <= x <= x2 and y1 <= y <= y2

    def process_frame(self, frame):
        results = self.model.track(frame, tracker="bytetrack.yaml", persist=True)
        boxes = results[0].boxes
        if boxes.id is not None:
            for i, box in enumerate(boxes.xyxy.cpu().numpy()):
                id = int(boxes.id[i])
                x1, y1, x2, y2 = map(int, box)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                self.tracked_ids.add(id)
                for zone in self.zones:
                    if self.inside_zone(cx, cy, zone):
                        self.zone_counts.setdefault(zone, set()).add(id)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, f"ID:{id}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        return frame, boxes

    def get_stats(self):
        return {
            "global_count": len(self.tracked_ids),
            "zone_counts": {k: len(v) for k, v in self.zone_counts.items()}
        }
