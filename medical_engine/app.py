import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import threading
import time
import urllib.request
import os
from flask import Flask, Response, jsonify, render_template, request
from pipeline.state import state, state_lock
from pipeline.gesture_engine import GestureEngine
from pipeline.template_engine import build_sentence, TOKENS as TOKEN_KB

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand landmarker model (~8MB)...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded.")


def open_camera():
    """Open the first available camera with Windows-friendly backend fallbacks."""
    attempts = [
        (0, cv2.CAP_DSHOW),
        (0, cv2.CAP_MSMF),
        (0, cv2.CAP_ANY),
        (1, cv2.CAP_DSHOW),
        (1, cv2.CAP_ANY),
    ]
    for index, backend in attempts:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            print(f"Camera opened: index={index}, backend={backend}")
            return cap
        cap.release()
    raise RuntimeError(
        "Could not open camera (tried indices 0/1 with DSHOW/MSMF/ANY). "
        "Close other apps using the webcam and verify camera permissions."
    )

gesture_engine = GestureEngine()

output_frame = None
frame_lock = threading.Lock()

latest_landmarks = []
landmarks_lock = threading.Lock()

VALID_TOKENS = set(TOKEN_KB.keys())


def landmarks_callback(result, output_image, timestamp_ms):
    with landmarks_lock:
        latest_landmarks.clear()
        if result.hand_landmarks:
            latest_landmarks.extend(result.hand_landmarks)


def camera_loop():
    global output_frame

    download_model()

    base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.LIVE_STREAM,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6,
        result_callback=landmarks_callback
    )

    timestamp = 0
    while True:
        cap = None
        try:
            cap = open_camera()
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            with mp_vision.HandLandmarker.create_from_options(options) as landmarker:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        time.sleep(0.05)
                        continue

                    frame = cv2.flip(frame, 1)
                    h, w = frame.shape[:2]

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                    timestamp += 1
                    landmarker.detect_async(mp_image, timestamp)

                    with landmarks_lock:
                        current_landmarks = list(latest_landmarks)

                    # Draw landmarks with confidence-aware colors.
                    connections = [
                        (0,1),(1,2),(2,3),(3,4),       # thumb
                        (0,5),(5,6),(6,7),(7,8),       # index
                        (5,9),(9,10),(10,11),(11,12),  # middle
                        (9,13),(13,14),(14,15),(15,16),# ring
                        (13,17),(0,17),(17,18),(18,19),(19,20) # pinky + palm
                    ]

                    with state_lock:
                        conf = state.get("confidence", 0)
                        active = state.get("active_gesture", "")

                    if conf >= 82:
                        skeleton_color = (74, 222, 128)    # green - locked
                    elif conf >= 50:
                        skeleton_color = (96, 165, 250)    # blue - detecting
                    elif active:
                        skeleton_color = (251, 191, 36)    # amber - weak
                    else:
                        skeleton_color = (180, 180, 180)   # gray - idle

                    for hand_lms in current_landmarks:
                        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms]
                        # Draw connections
                        for a, b in connections:
                            cv2.line(frame, pts[a], pts[b], skeleton_color, 2, cv2.LINE_AA)
                        # Draw joint dots; larger for tips.
                        tip_indices = {4, 8, 12, 16, 20}
                        for idx, pt in enumerate(pts):
                            r = 5 if idx in tip_indices else 3
                            cv2.circle(frame, pt, r, skeleton_color, -1, cv2.LINE_AA)
                            cv2.circle(frame, pt, r + 1, (0, 0, 0), 1, cv2.LINE_AA)

                    if conf > 20:
                        bar_w = int((conf / 100) * w)
                        bar_color = skeleton_color
                        cv2.rectangle(frame, (0, h - 4), (bar_w, h), bar_color, -1)
                        cv2.rectangle(frame, (0, h - 4), (w, h), (40, 40, 40), 1)

                    detected_token, confidence, scores = gesture_engine.process(current_landmarks, frame.shape)

                    with state_lock:
                        state["scores"] = scores
                        if detected_token and confidence > 0.30:
                            state["active_gesture"] = detected_token
                            state["confidence"] = round(confidence * 100)

                            if confidence >= gesture_engine.confirm_threshold:
                                existing = [t["name"] for t in state["tokens"]]
                                if not existing or existing[-1] != detected_token:
                                    state["tokens"].append({
                                        "name": detected_token,
                                        "source": "sign"
                                    })
                                    state["stage"] = 2
                                    gesture_engine.mark_committed(detected_token)  # per-token cooldown
                                    gesture_engine.reset()

                                    sentence = build_sentence([t["name"] for t in state["tokens"]])
                                    if sentence:
                                        state["sentence"] = sentence
                                        state["stage"] = 3
                        else:
                            state["active_gesture"] = ""
                            state["confidence"] = 0

                    with frame_lock:
                        output_frame = frame.copy()

                    time.sleep(0.033)
        except Exception as exc:
            print(f"Camera loop error: {exc}")
            time.sleep(1.0)
        finally:
            if cap is not None:
                cap.release()


def generate_frames():
    global output_frame
    while True:
        with frame_lock:
            if output_frame is None:
                time.sleep(0.05)
                continue
            ret, buffer = cv2.imencode(
                ".jpg", output_frame,
                [cv2.IMWRITE_JPEG_QUALITY, 70]
            )
            if not ret:
                continue
            frame_bytes = buffer.tobytes()

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            frame_bytes +
            b"\r\n"
        )
        time.sleep(0.04)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/video")
def video():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/data")
def data():
    with state_lock:
        return jsonify({
            "tokens": state["tokens"],
            "active_gesture": state["active_gesture"],
            "confidence": state["confidence"],
            "stage": state["stage"],
            "sentence": state["sentence"],
            "scores": state.get("scores", {})
        })


@app.route("/inject", methods=["POST"])
def inject():
    payload = request.get_json(silent=True) or {}
    token = payload.get("token", "").upper()
    source = payload.get("source", "tap")

    if token not in VALID_TOKENS:
        return jsonify(ok=False, error="unknown token"), 400

    with state_lock:
        state["tokens"].append({"name": token, "source": source})
        state["stage"] = 1

        sentence = build_sentence([t["name"] for t in state["tokens"]])
        if sentence:
            state["sentence"] = sentence
            state["stage"] = 3

    return jsonify(ok=True)


@app.route("/quick", methods=["POST"])
def quick():
    payload = request.get_json(silent=True) or {}
    sentence = payload.get("sentence", "")
    tokens = payload.get("tokens", [])

    if not sentence:
        return jsonify(ok=False), 400

    with state_lock:
        state["tokens"] = [{"name": t, "source": "quick"} for t in tokens]
        state["sentence"] = sentence
        state["stage"] = 3

    return jsonify(ok=True)


@app.route("/debug")
def debug():
    with state_lock:
        return jsonify({
            "tokens": state["tokens"],
            "active_gesture": state["active_gesture"],
            "confidence": state["confidence"],
            "stage": state["stage"],
            "sentence": state["sentence"],
            "scores": state.get("scores", {})
        })


@app.route("/submit", methods=["POST"])
def submit():
    """Force-lock the currently detected gesture regardless of confidence."""
    with state_lock:
        gesture = state["active_gesture"]
        if not gesture:
            return jsonify(ok=False, reason="no active gesture")
        existing = [t["name"] for t in state["tokens"]]
        if not existing or existing[-1] != gesture:
            state["tokens"].append({"name": gesture, "source": "sign"})
            state["stage"] = 2
            gesture_engine.mark_committed(gesture)
            gesture_engine.reset()
            sentence = build_sentence([t["name"] for t in state["tokens"]])
            if sentence:
                state["sentence"] = sentence
                state["stage"] = 3
    return jsonify(ok=True)

@app.route("/clear", methods=["POST"])
def clear():
    with state_lock:
        state["tokens"] = []
        state["active_gesture"] = ""
        state["confidence"] = 0
        state["stage"] = 0
        state["sentence"] = ""
    return jsonify(ok=True)


if __name__ == "__main__":
    cam_thread = threading.Thread(target=camera_loop, daemon=True)
    cam_thread.start()
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
