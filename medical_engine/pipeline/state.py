import threading

state_lock = threading.Lock()

state = {
    "tokens": [],
    "active_gesture": "",
    "confidence": 0,
    "stage": 0,
    "sentence": "",
    "scores": {}
}
