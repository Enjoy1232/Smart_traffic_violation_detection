from __future__ import annotations
import logging
from typing import Any

import numpy as np
from deep_sort_realtime.deepsort_tracker import DeepSort

import config

log = logging.getLogger(__name__)


class VehicleTracker:
    """Maintains a DeepSORT tracker and maps internal track IDs to integers."""

    def __init__(self) -> None:
        self.tracker = DeepSort(
            max_age=config.DEEPSORT_MAX_AGE,
            n_init=config.DEEPSORT_N_INIT,
            max_cosine_distance=config.DEEPSORT_MAX_COSINE_DISTANCE,
            nn_budget=None,
            override_track_class=None,
        )
        self._id_meta: dict[int, dict] = {}

    def update(
        self,
        vehicles: list[dict],
        frame: np.ndarray,
    ) -> list[dict[str, Any]]:
        """
        Update tracker with new vehicle detections.

        Parameters
        ----------
        vehicles : list of detection dicts from Detector.detect()
        frame    : BGR image (used by DeepSORT for appearance features)

        Returns
        -------
        list of confirmed-track dicts:
          {track_id, class_id, class_name, bbox:[x1,y1,x2,y2]}
        """
        if not vehicles:
            # Still update tracker to age-out lost tracks
            self.tracker.update_tracks([], frame=frame)
            return []

        raw_detections = []
        det_meta: list[dict] = []

        for det in vehicles:
            x1, y1, x2, y2 = det["bbox"]
            w = x2 - x1
            h = y2 - y1
            if w <= 0 or h <= 0:
                continue
            raw_detections.append(([x1, y1, w, h], det["conf"], det["class_name"]))
            det_meta.append({
                "class_id":   det["class_id"],
                "class_name": det["class_name"],
            })

        tracks = self.tracker.update_tracks(raw_detections, frame=frame)

        active_track_ids = {
            int(t.track_id) for t in tracks
            if not t.is_deleted() and t.is_confirmed()
        }
        stale_ids = set(self._id_meta.keys()) - active_track_ids
        for sid in stale_ids:
            self._id_meta.pop(sid, None)

        confirmed: list[dict] = []
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = int(track.track_id)
            det_class = track.det_class  # class_name string passed in

            if det_class is None and track_id in self._id_meta:
                det_class = self._id_meta[track_id]["class_name"]

            class_id = next(
                (cid for cid, name in config.VEHICLE_CLASS_IDS.items() if name == det_class),
                None,
            )
            if class_id is None:
                log.debug("Track %d has unknown class '%s', skipping.", track_id, det_class)
                continue

            # Cache metadata
            self._id_meta[track_id] = {
                "class_id":   class_id,
                "class_name": det_class or config.VEHICLE_CLASS_IDS.get(class_id, "Vehicle"),
            }

            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = (int(v) for v in ltrb)

            confirmed.append({
                "track_id":   track_id,
                "class_id":   class_id,
                "class_name": self._id_meta[track_id]["class_name"],
                "bbox":       [x1, y1, x2, y2],
            })

        return confirmed

    def get_active_ids(self) -> set[int]:
        """Returns IDs of tracks that are currently confirmed and not deleted."""
        return set(self._id_meta.keys())
