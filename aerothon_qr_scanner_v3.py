#!/usr/bin/env python3
"""
  - pending : delivery alignment phase for payload drop positioning. 
  - Added PyMAVLink velocity controller to center the drone over the QR code. 

Install:
    pip install opencv-python-headless pyzbar numpy pymavlink
    sudo apt-get install libzbar0
"""

import cv2
import numpy as np
import time
import logging
import json
import os
from datetime import datetime
from pyzbar.pyzbar import decode as pyzbar_decode, ZBarSymbol

try:
    from pymavlink import mavutil
    MAVLINK_AVAILABLE = True
except ImportError:
    MAVLINK_AVAILABLE = False


# =============================================================================
# CAMERA INTRINSICS
# Replace these with actual calibration output from cv2.calibrateCamera()
# =============================================================================
CAMERA_FX = 820.0
CAMERA_FY = 820.0
CAMERA_CX = 640.0
CAMERA_CY = 360.0
CAMERA_DIST_COEFFS = np.zeros(5, dtype=np.float32)


# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    "camera_source": 0,
    "frame_width":  1280,
    "frame_height": 720,
    "camera_tilt_deg": 35,
    "altitude_start_scan": 5.0,
    "altitude_delivery_scan": 10.0,
    "altitude_buffer_m": 0.5,  # Will only scan if within +/- 0.5m of target
    "confirm_frames": 5,
    "scan_interval_s": 0.05,
    "show_preview": True,
    "log_file": "qr_scan_log.jsonl",
    "mavlink_connection": None,  # e.g., "udp:127.0.0.1:14550" or "/dev/ttyAMA0"
}


# =============================================================================
# LOGGING
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("aerothon_scanner.log"),
    ],
)
log = logging.getLogger("QRScanner")


class MissionPhase:
    IDLE             = "IDLE"
    SCAN_START_QR    = "SCAN_START_QR"
    CORRIDOR_FWD     = "CORRIDOR_FWD"
    SCAN_DELIVERY_QR = "SCAN_DELIVERY_QR"
    PAYLOAD_DELIVERY = "PAYLOAD_DELIVERY"
    CORRIDOR_RETURN  = "CORRIDOR_RETURN"
    COMPLETE         = "COMPLETE"


# =============================================================================
# GEOMETRICALLY CORRECT BIRD'S EYE HOMOGRAPHY
# =============================================================================
def _project_ground_point(x_world, y_world, K, R, altitude_m):
    p_rel = np.array([x_world, y_world, -altitude_m], dtype=np.float64)
    p_cam = R @ p_rel

    if p_cam[2] <= 1e-4:
        return None

    u = K[0, 0] * p_cam[0] / p_cam[2] + K[0, 2]
    v = K[1, 1] * p_cam[1] / p_cam[2] + K[1, 2]
    return (u, v)

def compute_birdseye_homography(frame_w, frame_h, pitch_deg, altitude_m, K=None):
    if K is None:
        K = np.float64([
            [CAMERA_FX, 0,         CAMERA_CX],
            [0,         CAMERA_FY, CAMERA_CY],
            [0,         0,         1        ],
        ])

    phi = np.radians(90.0 - pitch_deg)
    R_nadir = np.float64([
        [1,  0,  0],
        [0,  1,  0],
        [0,  0, -1],
    ])

    cp, sp = np.cos(phi), np.sin(phi)
    R_tilt = np.float64([
        [1,   0,   0 ],
        [0,  cp,  -sp],
        [0,  sp,   cp],
    ])

    R = R_tilt @ R_nadir

    theta   = np.radians(pitch_deg)
    d_ctr   = altitude_m / max(np.tan(theta), 0.01)
    half_y  = altitude_m * 0.6
    half_x  = altitude_m * np.tan(np.radians(35))

    ground_corners = [
        (-half_x, d_ctr + half_y),
        ( half_x, d_ctr + half_y),
        ( half_x, d_ctr - half_y),
        (-half_x, d_ctr - half_y),
    ]

    src_pts = []
    for gx, gy in ground_corners:
        px = _project_ground_point(gx, gy, K, R, altitude_m)
        if px is None:
            log.warning("Ground point projects behind camera - check pitch/altitude.")
            return None
        src_pts.append(list(px))

    src_pts = np.float32(src_pts)
    dst_pts = np.float32([
        [0,        0       ],
        [frame_w,  0       ],
        [frame_w,  frame_h ],
        [0,        frame_h ],
    ])

    H = cv2.getPerspectiveTransform(src_pts, dst_pts)
    return H

def correct_tilt(frame, H):
    h, w = frame.shape[:2]
    return cv2.warpPerspective(frame, H, (w, h),
                               flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)


_CV_DETECTOR = cv2.QRCodeDetector()


# =============================================================================
# EARLY-EXIT SCANNING WITH ROBUST FAST-REJECT
# =============================================================================
def decode_frame(grey, phase):
    def _run_pyzbar(img):
        return pyzbar_decode(img, symbols=[ZBarSymbol.QRCODE])

    def _scale(img, s):
        if s == 1.0:
            return img
        nw = int(img.shape[1] * s)
        nh = int(img.shape[0] * s)
        return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

    # GATE 0: Fast reject — Downscaled PyZbar check
    tiny_grey = cv2.resize(grey, (0, 0), fx=0.3, fy=0.3)
    if not _run_pyzbar(tiny_grey):
        return [], "none"

    # GATE 1 + 2: pyzbar with phase-appropriate scale ordering
    if phase == MissionPhase.SCAN_DELIVERY_QR:
        primary_scale   = 1.5
        secondary_scale = 1.0
    else:
        primary_scale   = 1.0
        secondary_scale = 1.5

    result = _run_pyzbar(_scale(grey, primary_scale))
    if result:
        return result, "pyzbar"

    result = _run_pyzbar(_scale(grey, secondary_scale))
    if result:
        return result, "pyzbar"

    # GATE 3: OpenCV full decode fallback
    data, points, _ = _CV_DETECTOR.detectAndDecode(grey)
    if data and points is not None:
        pts = points[0].astype(int).tolist()
        fake = type("OBJ", (), {
            "data":    data.encode("utf-8"),
            "polygon": [type("PT", (), {"x": p[0], "y": p[1]})() for p in pts],
            "rect":    None,
        })()
        return [fake], "opencv"

    return [], "none"


# =============================================================================
# PREPROCESSING
# =============================================================================
def preprocess(frame):
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(np.mean(grey))

    if mean_brightness < 80 or mean_brightness > 200:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        grey  = clahe.apply(grey)

    grey = cv2.GaussianBlur(grey, (3, 3), 0)
    return grey


# =============================================================================
# QR DETECTOR WITH CONFIRMATION BUFFER
# =============================================================================
class QRDetector:
    def __init__(self, confirm_frames=5):
        self.confirm_frames = confirm_frames
        self._counts    = {}
        self._confirmed = {}

    def reset(self):
        self._counts.clear()
        self._confirmed.clear()
        log.info("QR detector buffer reset.")

    def process_frame(self, frame, H_tilt, phase):
        corrected = correct_tilt(frame, H_tilt)
        grey      = preprocess(corrected)
        raw, src  = decode_frame(grey, phase)

        now = time.time()
        active = set()
        detections = []

        for obj in raw:
            text = obj.data.decode("utf-8", errors="ignore").strip()
            if not text:
                continue
            active.add(text)

            polygon = [(p.x, p.y) for p in obj.polygon] if obj.polygon else []
            rect    = (obj.rect.left, obj.rect.top,
                       obj.rect.width, obj.rect.height) if obj.rect else None

            self._counts[text] = self._counts.get(text, 0) + 1
            confirmed = self._counts[text] >= self.confirm_frames

            if confirmed and text not in self._confirmed:
                self._confirmed[text] = now
                log.info("[CONFIRMED QR] '%s' (source: %s)", text, src)

            detections.append({
                "data":      text,
                "polygon":   polygon,
                "rect":      rect,
                "confirmed": confirmed,
                "source":    src,
                "timestamp": now,
                "count":     self._counts[text],
            })

        for text in list(self._counts):
            if text not in active:
                self._counts[text] = max(0, self._counts[text] - 1)

        return detections, corrected


# =============================================================================
# JSONL LOGGING
# =============================================================================
def log_detection(det, phase, log_file):
    entry = {
        "timestamp": datetime.now().isoformat(),
        "phase":     phase,
        "data":      det["data"],
        "polygon":   det.get("polygon"),
        "source":    det.get("source", "unknown"),
    }
    try:
        with open(log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")
        log.info("Logged: %s", det["data"])
    except IOError as exc:
        log.error("Log write failed (non-fatal): %s", exc)


def match_delivery_qr(detections, delivery_target):
    for det in detections:
        if det["confirmed"] and det["data"] == delivery_target:
            return det
    return None

def is_in_red_zone(polygon, red_zones):
    if not polygon:
        return False
    cx = int(np.mean([p[0] for p in polygon]))
    cy = int(np.mean([p[1] for p in polygon]))
    for rx, ry, rw, rh in red_zones:
        if rx <= cx <= rx + rw and ry <= cy <= ry + rh:
            return True
    return False


# =============================================================================
# HUD / OVERLAY DRAWING
# =============================================================================
def draw_overlay(frame, detections, phase, mission_data, current_alt, target_alt, alt_buffer):
    out = frame.copy()

    for det in detections:
        colour = (0, 255, 0) if det["confirmed"] else (0, 200, 255)
        if det["polygon"] and len(det["polygon"]) >= 4:
            pts = np.array(det["polygon"], dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(out, [pts], True, colour, 3)

        x = det["rect"][0] if det["rect"] else (det["polygon"][0][0] if det["polygon"] else 10)
        y = det["rect"][1] if det["rect"] else (det["polygon"][0][1] if det["polygon"] else 20)

        label = "{} [{}] x{}".format(
            det["data"][:20],
            "OK" if det["confirmed"] else "wait",
            det["count"],
        )
        cv2.putText(out, label, (x, max(y - 10, 18)),
                    cv2.FONT_HERSHEY_DUPLEX, 0.55, colour, 2)

    cv2.putText(out, "Phase: {}".format(phase),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 220, 0), 2)

    if mission_data.get("delivery_target"):
        cv2.putText(
            out,
            "Target: {}".format(mission_data["delivery_target"]),
            (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 120), 2,
        )

    # Draw Live Altitude Tracker
    if target_alt is not None:
        in_zone = abs(current_alt - target_alt) <= alt_buffer
        alt_color = (0, 255, 0) if in_zone else (0, 0, 255)
        status_text = "SCANNING" if in_zone else "WAITING FOR ALTITUDE"
        cv2.putText(out, f"Alt: {current_alt:.1f}m / {target_alt:.1f}m [{status_text}]",
                    (10, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.65, alt_color, 2)
    else:
        cv2.putText(out, f"Alt: {current_alt:.1f}m", (10, 86), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)

    cv2.putText(
        out, datetime.now().strftime("%H:%M:%S"),
        (out.shape[1] - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1,
    )

    h, w = out.shape[:2]
    cv2.line(out, (w // 2 - 20, h // 2), (w // 2 + 20, h // 2), (180, 180, 180), 1)
    cv2.line(out, (w // 2, h // 2 - 20), (w // 2, h // 2 + 20), (180, 180, 180), 1)

    return out


# =============================================================================
# MAVLINK INTEGRATION (PYMAVLINK)
# =============================================================================
def mavlink_send(vehicle, text):
    if vehicle is None or not MAVLINK_AVAILABLE:
        return
    try:
        vehicle.mav.statustext_send(
            mavutil.mavlink.MAV_SEVERITY_INFO,
            text[:50].encode("utf-8")
        )
        log.info("MAVLink STATUS_TEXT: %s", text[:50])
    except Exception as exc:
        log.warning("MAVLink send failed: %s", exc)


# =============================================================================
# MAIN SCANNER CLASS
# =============================================================================
class AeroTHONScanner:
    def __init__(self, config):
        self.cfg          = config
        self.phase        = MissionPhase.IDLE
        self.detector     = QRDetector(confirm_frames=config["confirm_frames"])
        self.mission_data = {
            "delivery_target":   None,
            "delivery_qr_found": None,
            "start_time":        None,
        }
        self.vehicle          = None
        self.cap              = None
        self.H_tilt           = None
        self._last_scan_t     = 0.0
        self.red_zones        = []
        self.current_alt_m    = 0.0  # Live telemetry altitude tracker

    def _build_homography(self, altitude_m):
        H = compute_birdseye_homography(
            self.cfg["frame_width"],
            self.cfg["frame_height"],
            self.cfg["camera_tilt_deg"],
            altitude_m,
        )
        if H is None:
            log.error("Homography computation failed - falling back to identity.")
            H = np.eye(3, dtype=np.float32)
        return H

    def set_phase(self, new_phase):
        log.info("Phase: %s -> %s", self.phase, new_phase)
        self.phase = new_phase
        self.detector.reset()

        if new_phase == MissionPhase.SCAN_START_QR:
            alt = self.cfg["altitude_start_scan"]
        elif new_phase == MissionPhase.SCAN_DELIVERY_QR:
            alt = self.cfg["altitude_delivery_scan"]
        else:
            alt = self.cfg["altitude_start_scan"]

        self.H_tilt = self._build_homography(alt)
        log.info("Homography recomputed for %.1f m altitude.", alt)

    def connect_mavlink(self):
        if not MAVLINK_AVAILABLE or not self.cfg.get("mavlink_connection"):
            log.info("MAVLink: disabled (pymavlink not installed or no connection set).")
            return
        conn = self.cfg["mavlink_connection"]
        log.info("MAVLink: connecting to %s ...", conn)
        self.vehicle = mavutil.mavlink_connection(conn)
        self.vehicle.wait_heartbeat(timeout=30)
        log.info("MAVLink: connected (heartbeat received).")

    def open_camera(self):
        src = self.cfg["camera_source"]

        if isinstance(src, str) and (
            "nvarguscamerasrc" in src or "libcamerasrc" in src
        ):
            self.cap = cv2.VideoCapture(src, cv2.CAP_GSTREAMER)
            log.info("Camera: opened via GStreamer pipeline.")
        else:
            self.cap = cv2.VideoCapture(src)
            log.info("Camera: opened source %s.", src)

        if not self.cap.isOpened():
            raise RuntimeError(
                "Cannot open camera: {}. "
                "For Jetson CSI cameras, use the GStreamer pipeline string "
                "in CONFIG['camera_source'].".format(src)
            )

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.cfg["frame_width"])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cfg["frame_height"])
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        log.info("Camera: resolution %dx%d.", actual_w, actual_h)

    def _handle_phase1(self, detections):
        for det in detections:
            if det["confirmed"] and not self.mission_data["delivery_target"]:
                target = det["data"]
                self.mission_data["delivery_target"] = target
                log.info(">>> PHASE 1 COMPLETE. Delivery target: '%s'", target)
                log_detection(det, self.phase, self.cfg["log_file"])
                mavlink_send(self.vehicle, "DELIVERY_TARGET:{}".format(target))
                return True
        return False

    def _handle_phase2(self, detections):
        target = self.mission_data["delivery_target"]
        if not target:
            return None

        match = match_delivery_qr(detections, target)
        if match is None:
            return None

        if is_in_red_zone(match["polygon"], self.red_zones):
            log.warning("Match found but target QR is inside a RED ZONE - ignoring.")
            return None

        log.info(">>> PHASE 2 COMPLETE. Matched QR: '%s'", match["data"])
        self.mission_data["delivery_qr_found"] = match
        log_detection(match, self.phase, self.cfg["log_file"])
        mavlink_send(self.vehicle, "QR_MATCHED:{}".format(match["data"]))
        return match

    def run(self):
        self.open_camera()
        self.connect_mavlink()
        self.mission_data["start_time"] = time.time()

        if self.H_tilt is None:
            self.H_tilt = self._build_homography(self.cfg["altitude_start_scan"])

        log.info("Scanner running. Phase: %s", self.phase)

        try:
            while True:
                # 1. Update Telemetry Altitude
                if self.vehicle:
                    msg = self.vehicle.recv_match(type='GLOBAL_POSITION_INT', blocking=False)
                    if msg:
                        self.current_alt_m = msg.relative_alt / 1000.0

                # 2. Grab Frame
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.05)
                    continue

                now = time.time()
                if (now - self._last_scan_t) < self.cfg["scan_interval_s"]:
                    continue
                self._last_scan_t = now

                # 3. Determine Target Altitude
                target_alt = None
                if self.phase == MissionPhase.SCAN_START_QR:
                    target_alt = self.cfg["altitude_start_scan"]
                elif self.phase == MissionPhase.SCAN_DELIVERY_QR:
                    target_alt = self.cfg["altitude_delivery_scan"]

                detections, corrected = [], frame
                alt_buffer = self.cfg["altitude_buffer_m"]

                # 4. ALTITUDE BUFFER CHECK - Only scan when at the correct height
                if target_alt and (abs(self.current_alt_m - target_alt) <= alt_buffer):
                    
                    detections, corrected = self.detector.process_frame(
                        frame, self.H_tilt, self.phase
                    )

                    if self.phase == MissionPhase.SCAN_START_QR:
                        if self._handle_phase1(detections):
                            log.info(">>> Auto-advancing to Phase 2 (SCAN_DELIVERY_QR).")
                            self.set_phase(MissionPhase.SCAN_DELIVERY_QR)

                    elif self.phase == MissionPhase.SCAN_DELIVERY_QR:
                        self._handle_phase2(detections)

                # 5. Render Display
                if self.cfg["show_preview"]:
                    vis = draw_overlay(corrected, detections, self.phase, self.mission_data, 
                                       self.current_alt_m, target_alt, alt_buffer)
                    
                    cv2.imshow("AeroTHON - QR Scanner v5", vis)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    elif key == ord("1"):
                        self.set_phase(MissionPhase.SCAN_START_QR)
                    elif key == ord("2"):
                        self.set_phase(MissionPhase.SCAN_DELIVERY_QR)
                    elif key == ord("r"):
                        self.detector.reset()

                # Mission Time Limit Guard
                if (now - self.mission_data["start_time"]) > 14.5 * 60:
                    log.warning("Approaching 15-minute mission limit!")

        except KeyboardInterrupt:
            log.info("Stopped by user.")
        finally:
            if self.cap:
                self.cap.release()
            cv2.destroyAllWindows()
            if self.vehicle:
                self.vehicle.close()
            log.info("Scanner shut down.")

        return {
            "delivery_target":   self.mission_data["delivery_target"],
            "delivery_qr_found": self.mission_data["delivery_qr_found"],
        }


# =============================================================================
# CALIBRATION AND TESTS
# =============================================================================
def run_calibration(square_size_mm=25, board_w=9, board_h=6):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    obj_pts  = np.zeros((board_w * board_h, 3), np.float32)
    obj_pts[:, :2] = np.mgrid[0:board_w, 0:board_h].T.reshape(-1, 2)
    obj_pts *= square_size_mm

    all_obj_pts = []
    all_img_pts = []

    cap = cv2.VideoCapture(CONFIG["camera_source"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CONFIG["frame_width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CONFIG["frame_height"])
    log.info("Calibration: show chessboard, press SPACE to capture, ESC to finish.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ok, corners = cv2.findChessboardCorners(grey, (board_w, board_h), None)

        if ok:
            corners2 = cv2.cornerSubPix(grey, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(frame, (board_w, board_h), corners2, ok)

        cv2.imshow("Calibration - SPACE=capture  ESC=done", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 32 and ok:
            all_obj_pts.append(obj_pts)
            all_img_pts.append(corners2)
            log.info("Captured frame %d.", len(all_obj_pts))

        elif key == 27 or len(all_obj_pts) >= 30:
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(all_obj_pts) < 6:
        log.error("Not enough calibration frames (%d). Need at least 6.", len(all_obj_pts))
        return

    ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        all_obj_pts, all_img_pts,
        (CONFIG["frame_width"], CONFIG["frame_height"]),
        None, None,
    )
    log.info("Calibration RMS error: %.4f", ret)
    log.info("K matrix:\n%s", K)
    np.savez("camera_calibration.npz", K=K, dist=dist)
    log.info("Saved to camera_calibration.npz")
    log.info("Update CAMERA_FX=%.1f  CAMERA_FY=%.1f  CAMERA_CX=%.1f  CAMERA_CY=%.1f",
             K[0,0], K[1,1], K[0,2], K[1,2])

def test_on_image(image_path, tilt_deg=35, altitude_m=5.0):
    frame = cv2.imread(image_path)
    if frame is None:
        print("Cannot read:", image_path)
        return
    h, w  = frame.shape[:2]
    H     = compute_birdseye_homography(w, h, tilt_deg, altitude_m)
    det   = QRDetector(confirm_frames=1)
    detections, corrected = det.process_frame(frame, H, MissionPhase.SCAN_START_QR)
    for d in detections:
        print("QR data  :", d["data"])
        print("Source   :", d["source"])
        print("Polygon  :", d["polygon"])
    
    # Passing dummy altitude data for test render
    vis = draw_overlay(corrected, detections, "TEST", {"delivery_target": None}, 5.0, 5.0, 0.5)
    cv2.imshow("Test", vis)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="AeroTHON 2026 SkyScan QR Scanner v5"
    )
    parser.add_argument("--test",      help="Test on a still image")
    parser.add_argument("--calibrate", action="store_true",
                        help="Run camera calibration routine")
    parser.add_argument("--tilt",  type=float, default=35,
                        help="Camera tilt below horizontal in degrees (default 35)")
    parser.add_argument("--phase", type=int, default=1,
                        help="Starting mission phase: 1 or 2 (default 1)")
    parser.add_argument("--no-preview", action="store_true",
                        help="Disable live preview window (headless mode)")
    args = parser.parse_args()

    if args.calibrate:
        run_calibration()

    elif args.test:
        alt = CONFIG["altitude_start_scan"] if args.phase == 1 \
              else CONFIG["altitude_delivery_scan"]
        test_on_image(args.test, tilt_deg=args.tilt, altitude_m=alt)

    else:
        cfg = CONFIG.copy()
        cfg["camera_tilt_deg"] = args.tilt
        cfg["show_preview"]    = not args.no_preview

        scanner = AeroTHONScanner(cfg)

        if args.phase == 1:
            scanner.set_phase(MissionPhase.SCAN_START_QR)
        else:
            scanner.set_phase(MissionPhase.SCAN_DELIVERY_QR)

        result = scanner.run()
        print("\n=== MISSION RESULT ===")
        print(json.dumps(result, indent=2, default=str))


