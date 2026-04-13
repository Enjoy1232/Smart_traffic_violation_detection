from __future__ import annotations
import logging
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Any

import numpy as np
import cv2
from ultralytics import YOLO

import config

log = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="yolo-submodel")


#Resolve YOLO device
def _resolve_device() -> str:
    """Return the best available device based on config.YOLO_DEVICE."""
    setting = str(config.YOLO_DEVICE).strip().lower()
    if setting == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return setting   # "cuda" or "cpu" — passed through as-is


#Detector

class Detector:
    """Loads YOLOv8 and runs per-frame inference."""

    def __init__(self) -> None:
        self._device = _resolve_device()
        log.info("Using device: %s", self._device.upper())

        self.model          = YOLO(config.MODEL_PATH)
        self.helmet_model   = YOLO(config.HELMET_MODEL_PATH)
        self.emergency_model = YOLO(
            getattr(config, "EMERGENCY_MODEL_PATH", "emergency_model.pt")
        )

        dummy = np.zeros((64, 64, 3), dtype=np.uint8)
        for mdl in (self.model, self.helmet_model, self.emergency_model):
            mdl(dummy, imgsz=config.YOLO_IMGSZ, device=self._device, verbose=False)

    def _run_helmet(self, inf_frame: np.ndarray) -> Any:
        """Run the helmet model (called from a thread-pool worker)."""
        return self.helmet_model(
            inf_frame,
            conf=getattr(config, "HELMET_CONF_THRESHOLD", 0.30),
            iou=config.YOLO_IOU_THRESHOLD,
            imgsz=config.YOLO_IMGSZ,
            device=self._device,
            verbose=False,
        )[0]

    def _run_emergency(self, inf_frame: np.ndarray) -> Any:
        """Run the emergency model (called from a thread-pool worker)."""
        return self.emergency_model(
            inf_frame,
            conf=getattr(config, "EMERGENCY_CONF_THRESHOLD", 0.40),
            iou=config.YOLO_IOU_THRESHOLD,
            imgsz=config.YOLO_IMGSZ,
            device=self._device,
            verbose=False,
        )[0]

    def detect(self, frame: np.ndarray) -> dict[str, Any]:
        """
        Run YOLOv8 on *frame* and return a structured result dict:

        {
          "vehicles":          list of {class_id, class_name, conf, bbox:[x1,y1,x2,y2]},
          "persons":           list of {conf, bbox:[x1,y1,x2,y2]},
          "traffic_lights":    list of {conf, bbox:[x1,y1,x2,y2]},
          "helmet_detections": list of {class_name, conf, bbox:[x1,y1,x2,y2]},
          "emergency_detections": list of {class_name, conf, bbox:[x1,y1,x2,y2]},
          "raw_dets":          list of {class_id, conf, bbox:[x1,y1,x2,y2]},
          "helmet_model_ran":  bool — True only when the helmet model was invoked.
        }

        bbox values are integer pixel coordinates in the ORIGINAL frame.
        """
        # Resize for inference
        orig_h, orig_w = frame.shape[:2]
        inf_w = getattr(config, "INFERENCE_INPUT_WIDTH",  640)
        inf_h = getattr(config, "INFERENCE_INPUT_HEIGHT", 640)
        if orig_w != inf_w or orig_h != inf_h:
            inf_frame = cv2.resize(frame, (inf_w, inf_h),
                                   interpolation=cv2.INTER_LINEAR)
        else:
            inf_frame = frame
        sx = orig_w / inf_w
        sy = orig_h / inf_h

        def _scale_bbox(box_xyxy):
            ix1, iy1, ix2, iy2 = box_xyxy
            return [
                max(0,        int(ix1 * sx)),
                max(0,        int(iy1 * sy)),
                min(orig_w,   int(ix2 * sx)),
                min(orig_h,   int(iy2 * sy)),
            ]

        results = self.model(
            inf_frame,
            conf=config.YOLO_CONF_THRESHOLD,
            iou=config.YOLO_IOU_THRESHOLD,
            imgsz=config.YOLO_IMGSZ,
            device=self._device,
            verbose=False,
        )[0]

        vehicles:       list[dict] = []
        persons:        list[dict] = []
        traffic_lights: list[dict] = []
        raw_dets:       list[dict] = []

        for box in results.boxes:
            class_id = int(box.cls[0])
            conf     = float(box.conf[0])
            bbox     = _scale_bbox(box.xyxy[0].tolist())

            raw_dets.append({"class_id": class_id, "conf": conf, "bbox": bbox})

            if class_id in config.VEHICLE_CLASS_IDS:
                vehicles.append({
                    "class_id":   class_id,
                    "class_name": config.VEHICLE_CLASS_IDS[class_id],
                    "conf":       conf,
                    "bbox":       bbox,
                })
            elif class_id == config.PERSON_CLASS_ID:
                persons.append({"conf": conf, "bbox": bbox})
            elif class_id == config.TRAFFIC_LIGHT_CLASS_ID:
                traffic_lights.append({"conf": conf, "bbox": bbox})

        has_motorcycles    = any(d["class_id"] in (1, 3) for d in vehicles)
        has_large_vehicles = any(d["class_id"] in (2, 5, 7) for d in vehicles)

        helmet_future:    Future | None = None
        emergency_future: Future | None = None

        if has_motorcycles:
            helmet_future = _executor.submit(self._run_helmet, inf_frame)
        if has_large_vehicles:
            emergency_future = _executor.submit(self._run_emergency, inf_frame)

        helmet_results    = helmet_future.result()    if helmet_future    else None
        emergency_results = emergency_future.result() if emergency_future else None

        helmet_detections: list[dict] = []
        if helmet_results is not None:
            for box in helmet_results.boxes:
                conf     = float(box.conf[0])
                class_id = int(box.cls[0])
                try:
                    class_name = helmet_results.names[class_id].lower()
                except Exception:
                    class_name = "helmet" if class_id == 0 else "no_helmet"
                helmet_detections.append({
                    "class_name": class_name,
                    "conf":       conf,
                    "bbox":       _scale_bbox(box.xyxy[0].tolist()),
                })

        emergency_detections: list[dict] = []
        if emergency_results is not None:
            for box in emergency_results.boxes:
                conf     = float(box.conf[0])
                class_id = int(box.cls[0])
                try:
                    class_name = emergency_results.names[class_id].lower()
                except Exception:
                    class_name = "emergency"
                emergency_detections.append({
                    "class_name": class_name,
                    "conf":       conf,
                    "bbox":       _scale_bbox(box.xyxy[0].tolist()),
                })

        return {
            "vehicles":             vehicles,
            "persons":              persons,
            "traffic_lights":       traffic_lights,
            "helmet_detections":    helmet_detections,
            "emergency_detections": emergency_detections,
            "raw_dets":             raw_dets,
            "helmet_model_ran":     helmet_results is not None,
        }
