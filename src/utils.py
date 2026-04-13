# utils.py — Drawing helpers, DB logging, and overlay utilities
"""
Provides:
  - draw_box()             : bounding box + label with colour
  - draw_stop_line()       : horizontal stop line
  - draw_signal_badge()    : top-right signal indicator
  - draw_status_bar()      : bottom info bar
  - init_db()              : initialize MySQL connection
  - log_violation_to_db()  : insert violation into MySQL (async)
  - reset_logged_violations(): clear per-session dedup set (call between videos)
"""

from __future__ import annotations
import atexit
import logging
import os
import queue
import threading
from datetime import datetime

import cv2
import numpy as np
import mysql.connector
from mysql.connector import Error

import config

# Canonical string for the emergency exemption violation type.
# Defined once here to avoid typo bugs across callers.
EXEMPTED_VIOLATION_TYPE: str = "Emergency Vehicle - Exempted"

log = logging.getLogger(__name__)
# NOTE: logging.basicConfig() is NOT called here.
# It is only configured once, in main.py's __main__ block.


# ─── Database connection (guarded by a lock) ──────────────────────────────────

_db_lock: threading.Lock = threading.Lock()
db_connection = None


def init_db() -> None:
    """Create the database/table if needed and open the shared connection."""
    from db_setup import setup_database
    setup_database()   # ensures DB and table exist

    global db_connection
    with _db_lock:
        try:
            if db_connection is None or not db_connection.is_connected():
                db_connection = mysql.connector.connect(
                    host=config.DB_HOST,
                    user=config.DB_USER,
                    password=config.DB_PASSWORD,
                    database=config.DB_NAME,
                )
                log.info("Connected to MySQL database '%s' successfully.", config.DB_NAME)
        except Exception as e:
            log.error("Could not connect to MySQL database: %s", e)
            db_connection = None


# ── Async DB queue (background thread) ────────────────────────────────────────

_db_queue: queue.Queue = queue.Queue()


def _db_worker() -> None:
    """Background thread: drain _db_queue until a None sentinel is received."""
    while True:
        job = _db_queue.get()
        if job is None:          # sentinel — shut down cleanly
            _db_queue.task_done()
            break
        _do_db_insert(*job)
        _db_queue.task_done()


_db_thread = threading.Thread(target=_db_worker, name="db-writer", daemon=True)
_db_thread.start()


def _shutdown_db_worker() -> None:
    """
    atexit handler: send the sentinel, wait up to 5 s for the worker to flush
    any remaining inserts, then return so Python can continue exiting.
    """
    _db_queue.put(None)
    _db_thread.join(timeout=5)


atexit.register(_shutdown_db_worker)


def _do_db_insert(
    track_id: int,
    vehicle_class: str,
    violation_type: str,
    plate_text: str,
    confidence: float,
    frame_num: int,
) -> None:
    """Perform the actual blocking MySQL insert (called from background thread)."""
    global db_connection
    is_exempted = bool(violation_type == EXEMPTED_VIOLATION_TYPE)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Reconnect if the connection has gone stale (lock protects the shared object)
    with _db_lock:
        try:
            if db_connection:
                db_connection.ping(reconnect=True, attempts=2, delay=1)
        except Exception:
            init_db()

        if db_connection and db_connection.is_connected():
            try:
                cursor = db_connection.cursor()
                query = """
                INSERT INTO violations
                (track_id, vehicle_class, violation_type, plate_number,
                 confidence, frame_number, timestamp, is_exempted)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """
                try:
                    cursor.execute(
                        query,
                        (
                            track_id, vehicle_class, violation_type, plate_text,
                            float(confidence) if confidence is not None else 0.0,
                            frame_num, ts, is_exempted,
                        ),
                    )
                    db_connection.commit()
                finally:
                    cursor.close()
            except Error as e:
                log.error("MySQL Insert Failed: %s", e)
        else:
            log.error("No database connection available to log violation.")


# ── Per-session deduplication set ─────────────────────────────────────────────

_logged_violations: set[tuple] = set()   # (track_id, violation_type)


def reset_logged_violations() -> None:
    """
    Clear the deduplication set.

    Must be called at the start of each new video so that a track_id seen in
    video 1 is not silently suppressed in video 2.
    """
    _logged_violations.clear()


def log_violation_to_db(
    track_id: int,
    vehicle_class: str,
    violation_type: str,
    plate_text: str,
    confidence: float,
    frame_num: int,
    deduplicate: bool = True,
) -> bool:
    """
    Queue a violation insert for the background DB writer thread.

    Returns True immediately if the violation is new (not a duplicate).
    The actual MySQL write happens asynchronously in _db_worker.
    """
    key = (track_id, violation_type)
    if deduplicate and key in _logged_violations:
        return False

    _logged_violations.add(key)
    _db_queue.put((track_id, vehicle_class, violation_type, plate_text, confidence, frame_num))
    return True


# ─── Drawing helpers ──────────────────────────────────────────────────────────

def draw_box(
    frame: np.ndarray,
    bbox: list[int],
    label: str,
    color: tuple[int, int, int],
) -> None:
    """Draw a bounding box with a filled label chip."""
    x1, y1, x2, y2 = bbox
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, config.BOX_THICKNESS)

    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = config.FONT_SCALE
    thickness  = config.FONT_THICKNESS
    pad        = 4

    def _chip(text: str, top_left: tuple[int, int], bg: tuple[int, int, int]) -> None:
        tw, th = cv2.getTextSize(text, font, font_scale, thickness)[0]
        bx1 = max(0, top_left[0])
        by1 = top_left[1] - th - pad * 2
        bx2 = bx1 + tw + pad * 2
        by2 = top_left[1]
        h, w = frame.shape[:2]
        by1 = max(0, by1)
        bx2 = min(w, bx2)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), bg, -1)
        cv2.putText(
            frame, text,
            (bx1 + pad, by2 - pad),
            font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA,
        )

    _chip(label, (x1, y1), color)


def draw_stop_line(frame: np.ndarray, stop_line_y: int) -> None:
    """Draw the fixed stop line as a dashed white horizontal line."""
    h, w = frame.shape[:2]
    y = max(0, min(stop_line_y, h - 1))
    dash_len = 30
    gap_len  = 15
    x = 0
    while x < w:
        cv2.line(frame, (x, y), (min(x + dash_len, w), y), (255, 255, 255), 2)
        x += dash_len + gap_len


def draw_signal_badge(frame: np.ndarray, signal_state: str) -> None:
    """Draw a coloured signal-state badge in the top-right corner."""
    color_map = {
        "RED":     (0,   0, 220),
        "YELLOW":  (0, 215, 255),
        "GREEN":   (0, 180,   0),
        "UNKNOWN": (80,  80,  80),
        "GREEN (no signal detected)": (0, 180, 0),
    }
    bg_color = color_map.get(signal_state, (80, 80, 80))
    h, w = frame.shape[:2]
    text  = f"SIGNAL: {signal_state}"
    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.6
    thick = 1
    tw, th = cv2.getTextSize(text, font, scale, thick)[0]
    pad   = 8
    x1 = w - tw - pad * 2 - 4
    y1 = 4
    x2 = w - 4
    y2 = th + pad * 2 + 4
    cv2.rectangle(frame, (x1, y1), (x2, y2), bg_color, -1)
    cv2.putText(
        frame, text,
        (x1 + pad, y2 - pad),
        font, scale, (255, 255, 255), thick, cv2.LINE_AA,
    )


def draw_status_bar(
    frame: np.ndarray,
    frame_num: int,
    fps: float,
    active_vehicles: int,
    total_violations: int,
) -> None:
    """Draw a dark info bar at the bottom of the frame."""
    h, w = frame.shape[:2]
    bar_h = 28
    cv2.rectangle(frame, (0, h - bar_h), (w, h), (20, 20, 20), -1)
    text = (
        f"FPS: {fps:.1f}  |  Frame: {frame_num}  |"
        f"  Vehicles: {active_vehicles}  |  Violations: {total_violations}"
    )
    cv2.putText(
        frame, text,
        (8, h - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA,
    )


def resize_for_display(frame: np.ndarray) -> np.ndarray:
    """Resize frame to the configured display dimensions."""
    return cv2.resize(
        frame,
        (config.DISPLAY_WIDTH, config.DISPLAY_HEIGHT),
        interpolation=cv2.INTER_LINEAR,
    )
