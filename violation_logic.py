from __future__ import annotations
import collections
import logging
import time

import numpy as np

import config

log = logging.getLogger(__name__)


class ViolationEngine:
    """
    Encapsulates all per-session violation state.

    One instance is created per FrameProcessor. Call reset() between videos
    to clear state while keeping the instance alive.
    """

    def __init__(self) -> None:
        self.violated_tracks:       set[int] = set()
        self.triple_riding_tracks:  set[int] = set()
        self.no_helmet_tracks:      set[int] = set()
        self.exempted_tracks:       set[int] = set()
        self._track_bboxes:         dict[int, list[int]] = {}
        self._recently_cleared_violators: dict[tuple[float, float], float] = {}
        self._no_helmet_buffer:     dict[int, int] = collections.defaultdict(int)
        self._emergency_class_map:  dict[int, str] = {}

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Clear all state for a new video."""
        self.violated_tracks.clear()
        self.triple_riding_tracks.clear()
        self.no_helmet_tracks.clear()
        self.exempted_tracks.clear()
        self._track_bboxes.clear()
        self._no_helmet_buffer.clear()
        self._emergency_class_map.clear()
        self._recently_cleared_violators.clear()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def get_emergency_class(self, track_id: int) -> str:
        return self._emergency_class_map.get(track_id, "EMERGENCY VEHICLE")

    def inherit_track(self, old_id: int, new_id: int) -> None:
        """Inherit persistent violation states and buffers from a recently lost track_id."""
        if old_id in self.violated_tracks:
            self.violated_tracks.add(new_id)
        if old_id in self.triple_riding_tracks:
            self.triple_riding_tracks.add(new_id)
        if old_id in self.no_helmet_tracks:
            self.no_helmet_tracks.add(new_id)
        if old_id in self._no_helmet_buffer:
            self._no_helmet_buffer[new_id] = self._no_helmet_buffer[old_id]
            self._no_helmet_buffer.pop(old_id, None)
        if old_id in self._emergency_class_map:
            self._emergency_class_map[new_id] = self._emergency_class_map[old_id]

    # ── Emergency Exemption ───────────────────────────────────────────────────

    def check_emergency_exemption(
        self,
        track_id: int,
        bbox: list[int],
        frame: np.ndarray,
        emergency_detections: list[dict],
        frame_num: int = 0,
    ) -> tuple[bool, str | None]:
        """
        Check if a track overlaps with an emergency vehicle detection.
        Returns (is_exempted, emergency_class_name).
        """
        if track_id in self.exempted_tracks:
            return True, self.get_emergency_class(track_id)

        v_area = max(1, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
        for ed in emergency_detections:
            ex1, ey1, ex2, ey2 = ed["bbox"]
            ix1 = max(bbox[0], ex1)
            iy1 = max(bbox[1], ey1)
            ix2 = min(bbox[2], ex2)
            iy2 = min(bbox[3], ey2)
            if ix1 < ix2 and iy1 < iy2:
                inter = (ix2 - ix1) * (iy2 - iy1)
                if (inter / v_area) > 0.75:
                    self.exempted_tracks.add(track_id)
                    emergency_class = ed["class_name"].upper()
                    self._emergency_class_map[track_id] = emergency_class
                    return True, emergency_class

        return False, None

    # ── 1. Signal Jump ────────────────────────────────────────────────────────

    def check_signal_jump(
        self,
        track_id: int,
        bbox: list[int],
        signal_state: str,
        stop_line_y: int,
        signal_detected: bool = True,
    ) -> str | None:
        """
        Returns "Signal Jump" if the vehicle's bottom edge is past the stop line
        while the signal is RED and a traffic signal is actually visible in frame.

        main.py performs emergency vehicle checks before ever calling
        check_signal_jump(), so no internal guard is needed here.
        """
        if not signal_detected:
            return None
        if signal_state != "RED":
            return None
        if track_id in self.violated_tracks:
            self._track_bboxes[track_id] = bbox
            return None

        bottom_y = bbox[3]
        if bottom_y > stop_line_y:
            now = time.monotonic()
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            grace = getattr(config, "ID_SWITCH_GRACE_PERIOD_SEC", 1.5)
            dist_thresh = getattr(config, "ID_SWITCH_MAX_DISTANCE", 70)

            skip_violation = False
            to_delete = []
            for (lcx, lcy), tstamp in list(self._recently_cleared_violators.items()):
                if now - tstamp > grace:
                    to_delete.append((lcx, lcy))
                else:
                    s_dist = ((cx - lcx) ** 2 + (cy - lcy) ** 2) ** 0.5
                    if s_dist < dist_thresh:
                        skip_violation = True
            for k in to_delete:
                self._recently_cleared_violators.pop(k, None)

            if skip_violation:
                return None

            self.violated_tracks.add(track_id)
            self._track_bboxes[track_id] = bbox
            return "Signal Jump"

        return None

    # ── 2. No Helmet ──────────────────────────────────────────────────────────

    def check_no_helmet(
        self,
        track_id: int,
        riders: list[dict],
        helmet_detections: list[dict],
        moto_bbox: list[int] = None,
        frame: np.ndarray = None,
        emergency_detections: list[dict] = None,
        frame_num: int = 0,
        helmet_model_ran: bool = True,
    ) -> str | None:
        """
        MOTORCYCLE-ONLY: returns "No Helmet" if any associated rider does not have
        an overlapping helmet detection in their head region.

        If helmet_model_ran is False (model was skipped this frame),
        the buffer is neither incremented nor reset — avoiding false violations
        caused by frames where the model simply didn't run.
        """
        if not config.HELMET_DETECTION_ENABLED:
            return None
        if track_id in self.no_helmet_tracks:
            return None

        # No riders on this motorcycle — nothing to evaluate
        if not riders:
            return None

        # Helmet model didn't run this frame — skip buffer update entirely
        if not helmet_model_ran:
            return None

        has_violation = False
        for rider in riders:
            rx1, ry1, rx2, ry2 = rider["bbox"]
            h = ry2 - ry1
            head_h = int(h * config.HEAD_REGION_FRACTION)
            head_bbox = [rx1, ry1, rx2, ry1 + head_h]
            head_area = max(1, (head_bbox[2] - head_bbox[0]) * (head_bbox[3] - head_bbox[1]))

            has_helmet = False
            for hd in helmet_detections:
                if hd["class_name"] == "helmet":
                    conf = float(hd.get("conf", 1.0))
                    if conf < getattr(config, "HELMET_MIN_CONFIDENCE", 0.40):
                        continue
                    hx1, hy1, hx2, hy2 = hd["bbox"]
                    ix1 = max(head_bbox[0], hx1)
                    iy1 = max(head_bbox[1], hy1)
                    ix2 = min(head_bbox[2], hx2)
                    iy2 = min(head_bbox[3], hy2)
                    if ix1 < ix2 and iy1 < iy2:
                        overlap_area = (ix2 - ix1) * (iy2 - iy1)
                        if (overlap_area / head_area) >= config.HELMET_OVERLAP_THRESHOLD:
                            has_helmet = True
                            break

            if not has_helmet:
                has_violation = True
                break

        if has_violation:
            self._no_helmet_buffer[track_id] += 1
            frames_threshold = getattr(config, "NO_HELMET_FRAMES_THRESHOLD", 3)
            if self._no_helmet_buffer[track_id] >= frames_threshold:
                self.no_helmet_tracks.add(track_id)
                return "No Helmet"
        else:
            # All riders confirmed with helmets — reset the buffer
            self._no_helmet_buffer[track_id] = 0

        return None

    # ── 3. Triple Riding ──────────────────────────────────────────────────────

    def check_triple_riding(
        self,
        track_id: int,
        moto_bbox: list[int],
        riders: list[dict],
        frame: np.ndarray = None,
        emergency_detections: list[dict] = None,
        frame_num: int = 0,
    ) -> str | None:
        """
        MOTORCYCLE-ONLY: counts confirmed riders on the motorcycle.

        Only counts persons whose horizontal center falls within the
        motorcycle's x-bounds (expanded by 25% of moto width) to avoid counting
        pedestrians standing beside the motorcycle.

        Violation: rider_count >= 3 → "Triple Riding"
        """
        if track_id in self.triple_riding_tracks:
            return None

        mx1, my1, mx2, my2 = moto_bbox
        moto_w = max(1, mx2 - mx1)
        x_min = mx1 - moto_w * 0.25
        x_max = mx2 + moto_w * 0.25

        confirmed_riders = [
            r for r in riders
            if x_min <= (r["bbox"][0] + r["bbox"][2]) / 2.0 <= x_max
        ]

        if len(confirmed_riders) >= 3:
            self.triple_riding_tracks.add(track_id)
            return "Triple Riding"

        return None

    # ── Cleanup ───────────────────────────────────────────────────────────────

    def clear_lost_track(self, track_id: int) -> None:
        """Remove all state for a track that has been lost / deleted by the tracker."""
        if track_id in self.violated_tracks and track_id in self._track_bboxes:
            bbox = self._track_bboxes.pop(track_id)
            cx = (bbox[0] + bbox[2]) / 2.0
            cy = (bbox[1] + bbox[3]) / 2.0
            self._recently_cleared_violators[(cx, cy)] = time.monotonic()

        self.violated_tracks.discard(track_id)
        self.triple_riding_tracks.discard(track_id)
        self.no_helmet_tracks.discard(track_id)
        self.exempted_tracks.discard(track_id)
        self._no_helmet_buffer.pop(track_id, None)
        self._emergency_class_map.pop(track_id, None)
