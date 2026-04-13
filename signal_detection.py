from __future__ import annotations
import logging
from collections import deque

import cv2
import numpy as np

import config

log = logging.getLogger(__name__)

_MAX_SIGNAL_ABSENT_FRAMES: int = 60


class SignalDetector:
    """
    Encapsulates all traffic signal detection state.

    One instance per FrameProcessor. Re-create between videos to get clean state.
    """

    def __init__(self) -> None:
        self._state_buffer: deque[tuple[str, bool]] = deque(maxlen=10)
        self._last_stable_state: str = "GREEN"
        self._last_tl_bbox: list[int] | None = None
        self._tl_missing_frames: int = 0
        self._signal_absent_frames: int = 0
        self._tl_classifier = self._load_classifier()


    def _load_classifier(self):
        """Load the YOLO traffic light classifier. Called once at construction."""
        try:
            from ultralytics import YOLO
            model = YOLO("traffic_light_classifier.pt")
            log.info("Traffic light classifier loaded.")
            return model
        except Exception as e:
            log.error("Could not load traffic light classifier: %s", e)
            return None

    def _analyse_crop_hsv(self, crop: np.ndarray) -> str:
        """
        HSV pixel-count fallback when the YOLO classifier is unavailable
        or returns UNKNOWN.

        Uses the colour ranges defined in config (SIGNAL_HSV_*).
        Returns 'RED', 'YELLOW', 'GREEN', or 'UNKNOWN'.
        """
        if crop.size == 0:
            return "UNKNOWN"

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        red_mask1 = cv2.inRange(
            hsv,
            np.array(config.SIGNAL_HSV_RED_LOWER1),
            np.array(config.SIGNAL_HSV_RED_UPPER1),
        )
        red_mask2 = cv2.inRange(
            hsv,
            np.array(config.SIGNAL_HSV_RED_LOWER2),
            np.array(config.SIGNAL_HSV_RED_UPPER2),
        )
        red_count = int(cv2.countNonZero(red_mask1) + cv2.countNonZero(red_mask2))

        yellow_mask = cv2.inRange(
            hsv,
            np.array(config.SIGNAL_HSV_YELLOW_LOWER),
            np.array(config.SIGNAL_HSV_YELLOW_UPPER),
        )
        yellow_count = int(cv2.countNonZero(yellow_mask))

        green_mask = cv2.inRange(
            hsv,
            np.array(config.SIGNAL_HSV_GREEN_LOWER),
            np.array(config.SIGNAL_HSV_GREEN_UPPER),
        )
        green_count = int(cv2.countNonZero(green_mask))

        min_pixels = getattr(config, "SIGNAL_MIN_PIXEL_COUNT", 200)
        counts = {"RED": red_count, "YELLOW": yellow_count, "GREEN": green_count}
        best_colour, best_count = max(counts.items(), key=lambda kv: kv[1])

        if best_count >= min_pixels:
            log.debug("HSV fallback: %s (%d px)", best_colour, best_count)
            return best_colour

        return "UNKNOWN"

    def _analyse_crop(self, crop: np.ndarray) -> str:
        """
        Classify a traffic light crop.

        1. Try the YOLO neural-net classifier.
        2. If classifier is unavailable or returns UNKNOWN, fall back to HSV.
        """
        nn_result = "UNKNOWN"
        if self._tl_classifier is not None:
            h, w = crop.shape[:2]
            min_size = getattr(config, "TRAFFIC_LIGHT_MIN_CROP_SIZE", 15)
            if h >= min_size and w >= min_size:
                work = crop.copy()
                # Upscale tiny crops so the classifier has enough detail
                if h < 64 or w < 64:
                    scale = max(64.0 / max(1, h), 64.0 / max(1, w))
                    work = cv2.resize(work, None, fx=scale, fy=scale,
                                      interpolation=cv2.INTER_CUBIC)
                work = cv2.normalize(work, None, alpha=0, beta=255,
                                     norm_type=cv2.NORM_MINMAX)
                results = self._tl_classifier(work, verbose=False)[0]
                if len(results.boxes) > 0:
                    conf = float(results.boxes[0].conf[0])
                    if conf >= getattr(config, "TRAFFIC_LIGHT_CLASSIFIER_CONF", 0.60):
                        class_id   = int(results.boxes[0].cls[0])
                        class_name = results.names[class_id].upper()
                        if class_name in ("RED", "YELLOW", "GREEN"):
                            nn_result = class_name

        if nn_result != "UNKNOWN":
            return nn_result

        # ── HSV fallback ──────────────────────────────────────────────────────
        return self._analyse_crop_hsv(crop)

    # ── Public API ────────────────────────────────────────────────────────────

    def detect_signal_color(
        self,
        frame: np.ndarray,
        traffic_lights: list[dict] | None = None,
    ) -> tuple[str, bool, int, list[int] | None]:
        """
        Detect the current traffic light state and the stop-line y-coordinate.

        Returns
        -------
        (stable_state, stable_detected, stop_line_y, best_tl_bbox)
        """
        yolo_lights  = traffic_lights or []
        best_tl_bbox = None

        # ── Pick the best YOLO-detected traffic light ──
        if yolo_lights:
            valid = [l for l in yolo_lights
                     if l["conf"] >= config.TRAFFIC_LIGHT_CONF_THRESHOLD]
            if valid:
                frame_cx = frame.shape[1] / 2.0

                def _sort_key(l):
                    x1, y1, x2, y2 = l["bbox"]
                    return (y1, abs((x1 + x2) / 2.0 - frame_cx))

                best = min(valid, key=_sort_key)
                best_tl_bbox = best["bbox"]
                self._last_tl_bbox = best_tl_bbox
                self._tl_missing_frames = 0

        # ── Carry forward last bbox for a few frames when YOLO misses ──
        if best_tl_bbox is None and self._last_tl_bbox is not None:
            if self._tl_missing_frames < getattr(
                    config, "TRAFFIC_LIGHT_TRACK_FRAMES", 5):
                self._tl_missing_frames += 1
                best_tl_bbox = self._last_tl_bbox

        # ── Classify the crop ──
        if best_tl_bbox is not None:
            x1, y1, x2, y2 = best_tl_bbox
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                state = self._analyse_crop(crop)
                if state != "UNKNOWN":
                    self._state_buffer.append((state, True))
                    self._signal_absent_frames = 0
                else:
                    self._state_buffer.append(("UNKNOWN", False))
                    self._signal_absent_frames += 1
            else:
                self._state_buffer.append(("UNKNOWN", False))
                self._signal_absent_frames += 1
        else:
            self._state_buffer.append(("UNKNOWN", False))
            self._signal_absent_frames += 1

        # ── Reset stale RED after too many absent frames ──────────────────────
        if self._signal_absent_frames >= _MAX_SIGNAL_ABSENT_FRAMES:
            log.debug(
                "Signal absent for %d consecutive frames — resetting to GREEN "
                "to prevent stale RED lock-in.",
                _MAX_SIGNAL_ABSENT_FRAMES,
            )
            self._last_stable_state = "GREEN"
            self._signal_absent_frames = 0

        # ── Stabilise signal presence (majority vote over buffer) ──
        recent_presence = [d for _, d in self._state_buffer]
        stable_detected = sum(recent_presence) >= (len(recent_presence) / 2.0)

        # ── Stabilise state using only confirmed-signal frames ──
        valid_states = [s for s, d in self._state_buffer if d]
        if valid_states:
            counts: dict[str, int] = {"RED": 0, "GREEN": 0, "YELLOW": 0, "UNKNOWN": 0}
            for s in valid_states:
                counts[s] = counts.get(s, 0) + 1
            stable_state = max(counts, key=lambda k: counts[k])
            if stable_state != "UNKNOWN":
                self._last_stable_state = stable_state
        else:
            stable_state = self._last_stable_state

        if stable_state == "UNKNOWN":
            stable_state = self._last_stable_state

        if not stable_detected:
            stable_state = self._last_stable_state

        stop_line_y = int(frame.shape[0] * config.STOP_LINE_Y_FRAC)
        return stable_state, stable_detected, stop_line_y, best_tl_bbox

    @staticmethod
    def is_red(signal_state: str) -> bool:
        """Convenience helper: True when the signal state is RED."""
        return signal_state == "RED"
