
import cv2
import csv
import math
import sys
import time
from datetime import datetime
from ultralytics import YOLO
from pythonosc import udp_client
from pythonosc import osc_message_builder

# OSC Setup
osc_ip = "127.0.0.1"
osc_port = 53000  # Max listens on this port
client = udp_client.UDPClient(osc_ip, osc_port)

# ---------------------------------------------------------------------------
# Multi-person settings
# ---------------------------------------------------------------------------
EXIT_GRACE_FRAMES = 15   # frames a track may vanish before /tracking/exit is sent
                         # (bridges brief occlusions so the voice doesn't cut out)

# Logging setup — writes a CSV file next to this script
log_filename = f"tracking_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
log_file = open(log_filename, "w", newline="")
csv_writer = csv.writer(log_file)
csv_writer.writerow(["timestamp", "num_people", "counter", "person_id",
                     "person_x", "person_y", "confidence", "bbox_w", "bbox_h",
                     "speed", "speed_x", "speed_y", "dispersion", "nearest", "event"])

def log_frame(num_people, counter, tracked, dispersion, nearest, event=""):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    if tracked:
        for i, (tid, info) in enumerate(tracked.items()):
            csv_writer.writerow([ts, num_people, counter, tid,
                                 f"{info['x']:.4f}", f"{info['y']:.4f}", f"{info['conf']:.4f}",
                                 f"{info['bw']:.4f}", f"{info['bh']:.4f}",
                                 f"{info['speed']:.4f}", f"{info['speed_x']:.4f}", f"{info['speed_y']:.4f}",
                                 f"{dispersion:.4f}", f"{nearest:.4f}",
                                 event if i == 0 else ""])
    else:
        csv_writer.writerow([ts, num_people, counter, "", "", "", "", "", "", "", "", "",
                             f"{dispersion:.4f}", f"{nearest:.4f}", event])
    log_file.flush()
    summary = " | ".join(
        f"id{tid}=({info['x']:.2f},{info['y']:.2f}) c={info['conf']:.2f} spd={info['speed']:.3f}"
        for tid, info in tracked.items()
    )
    if event:
        print(f"[{ts}] EVENT: {event} | people={num_people} counter={counter} | {summary}")
    else:
        print(f"[{ts}] people={num_people} counter={counter} disp={dispersion:.3f} near={nearest:.3f} | {summary}")

def send_osc(address, *args):
    msg = osc_message_builder.OscMessageBuilder(address=address)
    for arg in args:
        msg.add_arg(arg)
    client.send(msg.build())

# ---------------------------------------------------------------------------
# Load model & camera
# ---------------------------------------------------------------------------
model = YOLO("yolo26x.pt")
# 預設用攝影機；給一個影片路徑參數就改吃影片（方便單人測試多人情境）：
#   ./venv/bin/python app_multi.py ~/Downloads/people_walking.mp4
source = sys.argv[1] if len(sys.argv) > 1 else 0
webcamera = cv2.VideoCapture(source)
frame_w = webcamera.get(cv2.CAP_PROP_FRAME_WIDTH)
frame_h = webcamera.get(cv2.CAP_PROP_FRAME_HEIGHT)

# Persistent ID tracking is now done by Ultralytics' built-in ByteTrack
# (model.track, persist=True) — far more stable IDs than centroid matching.
tracks: dict = {}    # id -> {x, y, conf, bw, bh, time, speed_x, speed_y, speed}
missing: dict = {}   # id -> consecutive frames without a detection

counter                  = 0
absence_counter          = 0
person_present           = False
person_absent            = False
enter_threshold          = 3    # frames before triggering fade-up
exit_threshold           = 5    # frames before triggering QLab cues
position_reset_threshold = 10   # frames of absence before x/y/speed freeze back to 0

# "Last known" primary values — held across absence until reset threshold is hit
last_primary: dict = dict(x=0.0, y=0.0, conf=0.0, bw=0.0, bh=0.0,
                          speed=0.0, speed_x=0.0, speed_y=0.0)

# Max OSC address map:
#   /tracking/count         i  — number of people currently tracked
#   /tracking/dispersion    f  — mean distance of everyone to the group centroid
#                                (0 = everyone clustered together = harmony)
#   /tracking/nearest       f  — distance of the closest pair (meeting events)
#   /tracking/enter         i  — a new person id appeared
#   /tracking/exit          i  — a person id left (after grace period)
#   /tracking/<id>/x .. /bbox/h  — per-person data, ALL people, real ids
#   /tracking/x .. /present      — legacy primary-person messages (unchanged)

while True:
    now     = time.time()
    success, frame = webcamera.read()
    if not success:
        print("Failed to read from webcam")
        break

    # --- Detect + track people (conf ≥ 0.2, class 0 = person) ---------------
    results = model.track(frame, classes=0, conf=0.2, imgsz=480,
                          persist=True, verbose=False)
    boxes = results[0].boxes

    present_ids: set = set()
    if boxes is not None and boxes.id is not None:
        for i in range(len(boxes)):
            tid = int(boxes.id[i])
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            px = ((x1 + x2) / 2) / frame_w
            py = ((y1 + y2) / 2) / frame_h
            bw = (x2 - x1) / frame_w
            bh = (y2 - y1) / frame_h
            c  = float(boxes.conf[i])

            prev = tracks.get(tid)
            if prev is None:
                send_osc("/tracking/enter", tid)
                speed_x, speed_y, speed = 0.0, 0.0, 0.0
            else:
                dt = now - prev["time"]
                if dt > 0:
                    speed_x = (px - prev["x"]) / dt
                    speed_y = (py - prev["y"]) / dt
                    speed   = math.hypot(speed_x, speed_y)
                else:
                    speed_x, speed_y, speed = prev["speed_x"], prev["speed_y"], prev["speed"]

            tracks[tid] = dict(x=px, y=py, conf=c, bw=bw, bh=bh, time=now,
                               speed_x=speed_x, speed_y=speed_y, speed=speed)
            missing[tid] = 0
            present_ids.add(tid)

    # --- Exit detection: only after the grace period ---------------------------
    for tid in list(tracks.keys()):
        if tid not in present_ids:
            missing[tid] = missing.get(tid, 0) + 1
            if missing[tid] >= EXIT_GRACE_FRAMES:
                send_osc("/tracking/exit", tid)
                del tracks[tid]
                del missing[tid]

    # --- Per-person OSC (all people, real persistent ids) ----------------------
    for tid in present_ids:
        info = tracks[tid]
        send_osc(f"/tracking/{tid}/x",          float(info["x"]))
        send_osc(f"/tracking/{tid}/y",          float(info["y"]))
        send_osc(f"/tracking/{tid}/confidence", float(info["conf"]))
        send_osc(f"/tracking/{tid}/speed",      float(info["speed"]))
        send_osc(f"/tracking/{tid}/bbox/w",     float(info["bw"]))
        send_osc(f"/tracking/{tid}/bbox/h",     float(info["bh"]))

    # --- Group relation metrics -------------------------------------------------
    pts = [(tracks[tid]["x"], tracks[tid]["y"]) for tid in present_ids]
    n = len(pts)
    if n >= 2:
        cx = sum(p[0] for p in pts) / n
        cy = sum(p[1] for p in pts) / n
        dispersion = sum(math.hypot(p[0] - cx, p[1] - cy) for p in pts) / n
        nearest = min(math.hypot(a[0] - b[0], a[1] - b[1])
                      for i, a in enumerate(pts) for b in pts[i + 1:])
    else:
        dispersion = 0.0   # alone (or empty) counts as fully gathered = harmonic
        nearest = 1.0

    send_osc("/tracking/count",      n)
    send_osc("/tracking/dispersion", float(dispersion))
    send_osc("/tracking/nearest",    float(nearest))

    # --- Primary person (highest-confidence track) — legacy messages -----------
    num_people = n
    current_tracked = {tid: tracks[tid] for tid in present_ids}
    if current_tracked:
        primary      = max(current_tracked.values(), key=lambda d: d["conf"])
        last_primary = {k: primary[k] for k in
                        ("x", "y", "conf", "bw", "bh", "speed", "speed_x", "speed_y")}
        absence_counter = 0
    else:
        absence_counter += 1
        if absence_counter >= position_reset_threshold:
            last_primary = dict(x=0.0, y=0.0, conf=0.0, bw=0.0, bh=0.0,
                                speed=0.0, speed_x=0.0, speed_y=0.0)

    send_osc("/tracking/x",          float(last_primary["x"]))
    send_osc("/tracking/y",          float(last_primary["y"]))
    send_osc("/tracking/confidence", float(last_primary["conf"]))
    send_osc("/tracking/speed",      float(last_primary["speed"]))
    send_osc("/tracking/speed/x",    float(last_primary["speed_x"]))
    send_osc("/tracking/speed/y",    float(last_primary["speed_y"]))
    send_osc("/tracking/bbox/w",     float(last_primary["bw"]))
    send_osc("/tracking/bbox/h",     float(last_primary["bh"]))
    send_osc("/tracking/present",    1 if person_present else 0)

    event = ""

    if num_people > 0:
        counter += 1
        person_absent = False

        if counter == enter_threshold:
            event = "ENTER (cue 40)"
            send_osc("/cue/40/go")
            person_present = True

        if counter == exit_threshold:
            event = "STAY (cue 31)"
            send_osc("/cue/31/go")

    else:
        if person_present and not person_absent:
            event = "EXIT (cue 39+41)"
            send_osc("/cue/39/go")
            send_osc("/cue/41/go")
            person_absent = True
            person_present = False
        counter = 0

    log_frame(num_people, counter, current_tracked, dispersion, nearest, event)

    # --- Annotate and display ------------------------------------------------
    annotated = results[0].plot()
    info_text = (f"People: {num_people} | Disp: {dispersion:.3f} | Near: {nearest:.3f} "
                 f"| Counter: {counter}")
    cv2.putText(annotated, info_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.imshow("Live Camera", annotated)

    if cv2.waitKey(1) == ord("q"):
        break

webcamera.release()
cv2.destroyAllWindows()
log_file.close()
print(f"\nDetection stopped. Log saved to: {log_filename}")
