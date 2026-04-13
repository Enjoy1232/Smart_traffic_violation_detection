from __future__ import annotations
import collections
import logging
import os
import sys
import time

import cv2
import numpy as np

import config
import utils
from utils import EXEMPTED_VIOLATION_TYPE

from detector import Detector
from tracker import VehicleTracker
from signal_detection import SignalDetector
from violation_logic import ViolationEngine

log = logging.getLogger(__name__)


class FrameProcessor:
    """
    Encapsulates all per-frame state for the detection-tracking-violation pipeline.

    One instance is created at startup; call reset() between video/image files to
    clear violation history while keeping the models loaded.
    """

    _FPS_WINDOW: int = 30

    def __init__(self, detector: Detector, tracker: VehicleTracker) -> None:
        self.detector = detector
        self.tracker  = tracker

        # Per-frame timing
        self.frame_num:    int   = 0
        self.fps_display:  float = 25.0
        self._t_prev:      float = time.monotonic()
        self._frame_times: collections.deque[float] = collections.deque(
            maxlen=self._FPS_WINDOW
        )

        self._detect_interval: int = max(1, config.DETECT_EVERY_N_FRAMES)
        self._cached_detections: dict = {
            "vehicles": [], "persons": [], "traffic_lights": [],
            "raw_dets": [], "helmet_detections": [],
            "emergency_detections": [], "helmet_model_ran": False,
        }

        # Violation state and signal state are class instances
        self._violation_engine = ViolationEngine()
        self._signal_detector  = SignalDetector()

        # Per-vehicle bookkeeping 
        self._vehicle_violations:     dict[int, list[str]] = {}
        self._vehicle_last_violation: dict[int, str]       = {}
        self.total_violations:        int                  = 0
        self._lost_tracks_history:    dict[int, dict]      = {}
        self._last_bboxes:            dict[int, list[int]] = {}
        self._known_track_ids:        set[int]             = set()

        self._current_frame: np.ndarray | None = None


    def reset(self) -> None:
        """Reset violation state for a new video/image batch. Does NOT reset frame_num."""
        self._violation_engine.reset()
        self._signal_detector = SignalDetector()
        self.total_violations = 0
        self._vehicle_violations.clear()
        self._vehicle_last_violation.clear()
        self._lost_tracks_history.clear()
        self._last_bboxes.clear()
        self._known_track_ids.clear()
        self._frame_times.clear()
        self._current_frame = None
        self._cached_detections = {
            "vehicles": [], "persons": [], "traffic_lights": [],
            "raw_dets": [], "helmet_detections": [],
            "emergency_detections": [], "helmet_model_ran": False,
        }
        utils.reset_logged_violations()

    def process(self, frame: np.ndarray) -> np.ndarray:
        """Run the full pipeline on one frame and return the annotated display frame."""
        self.frame_num += 1

        self._current_frame = frame

        # ── FPS calculation (rolling average) ──
        t_now = time.monotonic()
        elapsed = t_now - self._t_prev
        self._t_prev = t_now
        if elapsed > 0:
            self._frame_times.append(elapsed)
        if self._frame_times:
            avg = sum(self._frame_times) / len(self._frame_times)
            self.fps_display = 1.0 / avg if avg > 0 else self.fps_display

        # ── 1. Object detection 
        is_detection_frame = (self.frame_num % self._detect_interval == 0)
        if is_detection_frame:
            self._cached_detections = self.detector.detect(frame)

        vehicles             = self._cached_detections["vehicles"]
        persons              = self._cached_detections["persons"]
        traffic_lights       = self._cached_detections["traffic_lights"]
        helmet_detections    = self._cached_detections.get("helmet_detections", [])
        emergency_detections = self._cached_detections.get("emergency_detections", [])
        helmet_model_ran     = self._cached_detections.get("helmet_model_ran", False)

        # ── 2. Vehicle tracking ──
        if is_detection_frame:
            tracks = self.tracker.update(vehicles, frame)
        else:
            tracks = self.tracker.update([], frame)

        # ── 3. Signal detection ──
        signal_state, signal_detected, stop_line_y, best_tl_bbox = \
            self._signal_detector.detect_signal_color(frame, traffic_lights)

        # ── 4. Draw static overlays ──
        if signal_detected:
            utils.draw_stop_line(frame, stop_line_y)

        if best_tl_bbox is not None:
            tl_color_map = {
                "RED":     (0, 0, 255),
                "YELLOW":  (0, 255, 255),
                "UNKNOWN": (128, 128, 128),
            }
            tl_color = tl_color_map.get(signal_state, (0, 255, 0))
            utils.draw_box(frame, best_tl_bbox,
                           f"TRAFFIC SIGNAL: {signal_state}", tl_color)

        # ── 5. Pre-calculate motorcycle → rider assignments
        motorcycles = [t for t in tracks if t["class_id"] in (1, 3)]
        moto_riders = self._match_riders(motorcycles, persons, frame)

        # ── 6. Per-vehicle violation logic + drawing
        active_ids_this_frame: set[int] = set()

        for track in tracks:
            tid        = track["track_id"]
            class_id   = track["class_id"]
            class_name = track["class_name"]
            bbox       = track["bbox"]
            active_ids_this_frame.add(tid)
            self._known_track_ids.add(tid)

            # ── Emergency vehicle check ──
            is_emergency_track = False
            if class_id not in (1, 3) and emergency_detections:
                is_emergency_track, emerg_raw = \
                    self._violation_engine.check_emergency_exemption(
                        tid, bbox, frame, emergency_detections, self.frame_num
                    )
                if is_emergency_track and emerg_raw:
                    raw = emerg_raw.lower()
                    if "ambulance" in raw and "fire" in raw:
                        class_name = "AMBULANCE/FIRE BRIGADE"
                    elif "fire" in raw:
                        class_name = "FIRE BRIGADE"
                    elif "ambulance" in raw:
                        class_name = "AMBULANCE"
                    elif "police" in raw:
                        class_name = "POLICE"
                    else:
                        class_name = "EMERGENCY VEHICLE"

            if is_emergency_track:
                utils.draw_box(frame, bbox,
                               f"ID {tid} - {class_name} - EXEMPTED",
                               (0, 255, 255))
                continue

            self._last_bboxes[tid] = bbox

            # ── Handle DeepSORT ID switches ──
            if tid not in self._vehicle_violations:
                self._vehicle_violations[tid] = []
                self._try_inherit_from_lost(tid, bbox)

            # ── Signal Jump ──
            sj = self._violation_engine.check_signal_jump(
                tid, bbox, signal_state, stop_line_y, signal_detected
            )
            if sj:
                self._add_violation(tid, class_name, sj, bbox,
                                    track.get("conf", 0.0))

            # ── Motorcycle-specific checks ──
            if class_id in (1, 3):
                riders = moto_riders.get(tid, [])

                tr = self._violation_engine.check_triple_riding(
                    tid, bbox, riders, frame, emergency_detections, self.frame_num
                )
                if tr:
                    self._add_violation(tid, class_name, tr, bbox,
                                        track.get("conf", 0.0))

                nh = self._violation_engine.check_no_helmet(
                    tid, riders, helmet_detections, bbox, frame,
                    emergency_detections, self.frame_num,
                    helmet_model_ran=helmet_model_ran,
                )
                if nh:
                    self._add_violation(tid, class_name, nh, bbox,
                                        track.get("conf", 0.0))

            # ── Resolve display colour & label ──
            if class_id == 1:
                class_name = "Motorcycle"

            current_viols = self._vehicle_violations.get(tid, [])
            viol_label = None
            if "Signal Jump"    in current_viols: viol_label = "SIGNAL JUMP"
            elif "Triple Riding" in current_viols: viol_label = "TRIPLE RIDING"
            elif "No Helmet"     in current_viols: viol_label = "NO HELMET"

            if viol_label is None:
                box_color = (0, 255, 0)
                label = f"ID {tid} - {class_name.upper()}"
            else:
                box_color = (0, 0, 255)
                label = f"ID {tid} - {class_name.upper()} - {viol_label}"

            if tid in self._violation_engine.exempted_tracks \
                    and class_id not in (1, 3):
                box_color = (0, 255, 255)
                label = f"ID {tid} - {class_name.upper()} - EXEMPTED"

            utils.draw_box(frame, bbox, label, box_color)

        # ── Helmet debug boxes ──
        for hd in helmet_detections:
            if hd["class_name"] in ("no_helmet", "without helmet"):
                utils.draw_box(frame, hd["bbox"],
                               f"WITHOUT HELMET {hd['conf']:.2f}", (0, 0, 255))

        # ── 7. Clean up state for disappeared tracks ──
        all_active = self.tracker.get_active_ids()
        lost_ids   = self._known_track_ids - all_active
        now_time   = time.monotonic()

        for lid in lost_ids:
            if lid in self._last_bboxes:
                self._lost_tracks_history[lid] = {
                    "bbox":       self._last_bboxes[lid],
                    "time":       now_time,
                    "violations": self._vehicle_violations.get(lid, []).copy(),
                }
            self._violation_engine.clear_lost_track(lid)
            self._vehicle_violations.pop(lid, None)
            self._last_bboxes.pop(lid, None)
            self._known_track_ids.discard(lid)

        # ── 8. Signal badge + status bar ──
        utils.draw_signal_badge(frame, signal_state)
        utils.draw_status_bar(
            frame, self.frame_num, self.fps_display,
            len(active_ids_this_frame), self.total_violations,
        )

        # ── 9. Resize for display ──
        return utils.resize_for_display(frame)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _add_violation(
        self,
        track_id: int,
        vehicle_type: str,
        violation_type: str,
        bbox: list[int],
        confidence: float = 0.0,
    ) -> None:
        """
        Record a violation.

        """
        # Identification is based strictly on Tracking ID
        plate_text = f"TRACK-{track_id}"

        logged = utils.log_violation_to_db(
            track_id, vehicle_type, violation_type,
            plate_text, confidence, self.frame_num
        )
        if logged:
            if violation_type != EXEMPTED_VIOLATION_TYPE:
                self.total_violations += 1
            self._vehicle_violations.setdefault(track_id, []).append(violation_type)
            self._vehicle_last_violation[track_id] = violation_type
            log.info("VIOLATION: Track %d — %s (plate: %s)", track_id, violation_type, plate_text)
        else:
            log.warning("Duplicate or DB error: Track %d — %s",
                        track_id, violation_type)

    def _try_inherit_from_lost(self, tid: int, bbox: list[int]) -> None:
        """Inherit violation history from a recently lost track at the same location."""
        tx1, ty1, tx2, ty2 = bbox
        t_center = np.array([(tx1 + tx2) / 2.0, (ty1 + ty2) / 2.0])
        now_time = time.monotonic()
        grace    = getattr(config, "ID_SWITCH_GRACE_PERIOD_SEC", 1.5)
        max_dist = getattr(config, "ID_SWITCH_MAX_DISTANCE", 70)

        best_match_id: int | None = None
        best_dist = float("inf")

        for lost_id, lost_info in list(self._lost_tracks_history.items()):
            if now_time - lost_info["time"] > grace:
                del self._lost_tracks_history[lost_id]
                continue
            lx1, ly1, lx2, ly2 = lost_info["bbox"]
            l_center = np.array([(lx1 + lx2) / 2.0, (ly1 + ly2) / 2.0])
            s_dist   = float(np.linalg.norm(t_center - l_center))
            if s_dist < max_dist and s_dist < best_dist:
                best_dist     = s_dist
                best_match_id = lost_id

        if best_match_id is not None:
            inherited = self._lost_tracks_history[best_match_id]["violations"]
            self._vehicle_violations[tid].extend(inherited)
            if inherited:
                self._vehicle_last_violation[tid] = inherited[-1]
            self._violation_engine.inherit_track(best_match_id, tid)
            del self._lost_tracks_history[best_match_id]

    def _match_riders(
        self,
        motorcycles: list[dict],
        persons: list[dict],
        frame: np.ndarray,
    ) -> dict[int, list[dict]]:
        """
        Assign each person to their closest motorcycle using vectorised numpy
        distance calculation + vertical / horizontal overlap constraints.

        Returns {track_id: [person_dict, ...]}
        """
        moto_riders: dict[int, list[dict]] = {m["track_id"]: [] for m in motorcycles}
        if not motorcycles or not persons:
            return moto_riders

        moto_centers = np.array(
            [[(m["bbox"][0] + m["bbox"][2]) / 2.0,
              (m["bbox"][1] + m["bbox"][3]) / 2.0]
             for m in motorcycles],
            dtype=float,
        )

        max_dist_px    = frame.shape[1] * 0.20
        rider_slack_px = int(frame.shape[0] *
                             getattr(config, "RIDER_VERTICAL_SLACK_FRAC", 0.08))

        for p in persons:
            px1, py1, px2, py2 = p["bbox"]
            p_center = np.array([(px1 + px2) / 2.0, (py1 + py2) / 2.0],
                                 dtype=float)

            dists = np.linalg.norm(moto_centers - p_center, axis=1)

            best_dist: float      = float("inf")
            best_tid:  int | None = None

            for i, dist in enumerate(dists):
                if dist >= max_dist_px or dist >= best_dist:
                    continue
                m = motorcycles[i]
                mx1, my1, mx2, my2 = m["bbox"]

                if py2 < my1 - rider_slack_px:
                    continue

                person_w = px2 - px1
                if person_w > 0:
                    overlap_w = max(0, min(px2, mx2) - max(px1, mx1))
                    if (overlap_w / person_w) > 0.20:
                        best_dist = dist
                        best_tid  = m["track_id"]

            if best_tid is not None:
                moto_riders[best_tid].append(p)

        return moto_riders


# ─── Video mode helper ────────────────────────────────────────────────────────

def _run_video_mode(processor: FrameProcessor) -> None:
    """Process all video files in config.VIDEO_FOLDER."""
    video_folder = getattr(config, "VIDEO_FOLDER", "videos")
    if not os.path.exists(video_folder):
        log.error("Video folder not found: %s", video_folder)
        sys.exit(1)

    video_files = sorted(
        f for f in os.listdir(video_folder)
        if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm"))
    )
    if not video_files:
        log.error("No videos found in '%s'", video_folder)
        sys.exit(1)

    log.info("Mode: VIDEO | Folder: %s | Files: %d", video_folder, len(video_files))
    quit_all = False
    paused   = False

    for video_name in video_files:
        if quit_all:
            break

        video_path = os.path.join(video_folder, video_name)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            log.warning("Cannot open video: %s — skipping", video_path)
            continue

        fps_source   = cap.get(cv2.CAP_PROP_FPS) or 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        log.info("Processing: %s | FPS: %.1f | Frames: %d",
                 video_name, fps_source, total_frames)

        processor.reset()
        processor.fps_display = fps_source

        writer: cv2.VideoWriter | None = None
        writer_initialized = False
        out_path = ""
        if getattr(config, "SAVE_OUTPUT_VIDEO", False):
            out_dir = getattr(config, "OUTPUT_VIDEO_DIR", "output_videos")
            os.makedirs(out_dir, exist_ok=True)
            base_name = os.path.splitext(video_name)[0]
            out_path  = os.path.join(out_dir, f"{base_name}_annotated.mp4")

        while True:
            if not paused:
                ret, frame = cap.read()
                if not ret:
                    break

                display_frame = processor.process(frame)

                if getattr(config, "SAVE_OUTPUT_VIDEO", False) \
                        and not writer_initialized:
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    h, w   = display_frame.shape[:2]
                    writer = cv2.VideoWriter(out_path, fourcc, fps_source, (w, h))
                    writer_initialized = True

                if writer is not None:
                    writer.write(display_frame)

                cv2.imshow(config.WINDOW_NAME, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                log.info("Quit requested.")
                quit_all = True
                break
            elif key == ord("r"):
                paused = not paused
                log.info("%s", "Paused" if paused else "Resumed")
            elif key == ord("n"):
                log.info("Skipping to next video…")
                break

        cap.release()
        if writer is not None:
            writer.release()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    mode = config.MODE

    utils.init_db()
    detector  = Detector()
    tracker   = VehicleTracker()
    processor = FrameProcessor(detector, tracker)
    log.info("YOLOv8 models loaded. Starting processing… (mode=%s)", mode)

    if mode == "video":
        _run_video_mode(processor)
    else:
        log.error("Unknown MODE '%s' in config.py. Use 'video'.", mode)
        sys.exit(1)

    cv2.destroyAllWindows()
    log.info("Done. Processed %d frames total.", processor.frame_num)
    log.info("Total violations logged: %d", processor.total_violations)
    log.info("Violations saved to: MySQL Database")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # basicConfig is configured ONLY here — never inside library modules
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    main()
