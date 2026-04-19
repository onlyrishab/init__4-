import math
from collections import deque


class GestureEngine:
    def __init__(self, buffer_size=20, confirm_frames=15, confirm_threshold=0.82):
        self.buffer_size = buffer_size
        self.confirm_frames = confirm_frames
        self.confirm_threshold = confirm_threshold

        self.tokens = [
            "PAIN", "FEVER", "NAUSEA", "HEAD", "CHEST",
            "STOMACH", "YES", "NO", "HELP", "STOP",
            "WATER", "COUGH", "DIZZY", "THROAT", "EYE",
            "THANK", "DOCTOR", "SEVERE", "MEDICINE", "BATHROOM"
        ]
        self.confidence_buffers = {t: deque(maxlen=buffer_size) for t in self.tokens}
        self.velocity_history = deque(maxlen=20)
        self.prev_wrist = None
        self.cooldown = 0

    def reset(self):
        for buf in self.confidence_buffers.values():
            buf.clear()
        self.velocity_history.clear()
        self.prev_wrist = None
        self.cooldown = 15

    def process(self, hand_landmarks_list, frame_shape):
        if self.cooldown > 0:
            self.cooldown -= 1
            return None, 0.0, {}

        if not hand_landmarks_list:
            for buf in self.confidence_buffers.values():
                buf.append(0.0)
            self.prev_wrist = None
            return None, 0.0, {}

        h, w = frame_shape[:2]
        hand = hand_landmarks_list[0]
        # Support both old mediapipe (.landmark) and new Tasks API (plain list)
        lm = hand.landmark if hasattr(hand, 'landmark') else hand

        # ── HAND VISIBILITY GUARD ───────────────────────────────────────────
        # Reject detections where the hand is too small (likely background noise)
        # or too far from center (edge artifacts)
        hand_size = self._hand_size(lm, h, w)
        if hand_size < 0.05:  # hand bounding box < 5% of frame → too small / noise
            for buf in self.confidence_buffers.values():
                buf.append(0.0)
            return None, 0.0, {}

        norm = self._normalize(lm)
        fingers = self._finger_states(norm)

        wrist_px = (lm[0].x * w, lm[0].y * h)
        if self.prev_wrist:
            vx = wrist_px[0] - self.prev_wrist[0]
            vy = wrist_px[1] - self.prev_wrist[1]
        else:
            vx, vy = 0.0, 0.0
        self.prev_wrist = wrist_px
        self.velocity_history.append((vx, vy))

        # ── STILLNESS CHECK ─────────────────────────────────────────────────
        # Many false positives come from hands moving → require relative stillness
        # for static gestures (STOP, HELP, PAIN, etc.)
        still = self._is_still()

        scores = {
            "STOP":    self._rule_stop(fingers, still),
            "HELP":    self._rule_help(fingers, still),
            "YES":     self._rule_yes(fingers),
            "NO":      self._rule_no(fingers),
            "HEAD":    self._rule_head(lm, h, fingers, still),
            "FEVER":   self._rule_fever(lm, h, fingers, still),
            "CHEST":   self._rule_chest(lm, h, w, fingers, still),
            "PAIN":    self._rule_pain(lm, h, w, fingers, still),
            "STOMACH": self._rule_stomach(lm, h, fingers, still),
            "NAUSEA":  self._rule_nausea(lm, h, fingers),
            "WATER":   self._rule_water(lm, h, fingers, still),
            "COUGH":   self._rule_cough(lm, h, w, fingers, still),
            "DIZZY":   self._rule_dizzy(lm, h, fingers),
            "THROAT":  self._rule_throat(lm, h, w, fingers, still),
            "EYE":     self._rule_eye(lm, h, fingers, still),
            "THANK":   self._rule_thank(fingers),
            "DOCTOR":  self._rule_doctor(lm, h, fingers, still),
            "SEVERE":  self._rule_severe(fingers),
            "MEDICINE":self._rule_medicine(lm, h, fingers, still),
            "BATHROOM":self._rule_bathroom(fingers),
        }

        # ── MUTUAL EXCLUSION ────────────────────────────────────────────────
        # If multiple static palm gestures fire, only keep the best one
        # This prevents STOP/FEVER/CHEST/STOMACH from all triggering at once
        palm_gestures = ["STOP", "FEVER", "CHEST", "STOMACH"]
        palm_scores = {g: scores[g] for g in palm_gestures if scores[g] > 0}
        if len(palm_scores) > 1:
            best_palm = max(palm_scores, key=palm_scores.get)
            for g in palm_gestures:
                if g != best_palm:
                    scores[g] = 0.0

        # Same for fist gestures (PAIN vs YES vs HELP vs COUGH)
        fist_gestures = ["PAIN", "YES", "HELP", "COUGH"]
        fist_scores = {g: scores[g] for g in fist_gestures if scores[g] > 0}
        if len(fist_scores) > 1:
            best_fist = max(fist_scores, key=fist_scores.get)
            for g in fist_gestures:
                if g != best_fist:
                    scores[g] = 0.0

        # Same for index finger gestures (HEAD vs EYE vs THROAT)
        idx_gestures = ["HEAD", "EYE", "THROAT"]
        idx_scores = {g: scores[g] for g in idx_gestures if scores[g] > 0}
        if len(idx_scores) > 1:
            best_idx = max(idx_scores, key=idx_scores.get)
            for g in idx_gestures:
                if g != best_idx:
                    scores[g] = 0.0

        for token, score in scores.items():
            self.confidence_buffers[token].append(float(score))

        best_token = None
        best_conf = 0.0
        avg_scores = self._compute_avg_scores()

        for token, avg in avg_scores.items():
            if avg > best_conf:
                best_conf = avg
                best_token = token

        if best_conf >= self.confirm_threshold:
            return best_token, best_conf, avg_scores
        if best_conf > 0.30:
            return best_token, best_conf, avg_scores
        return None, 0.0, avg_scores

    def _compute_avg_scores(self):
        out = {}
        for token, buf in self.confidence_buffers.items():
            if len(buf) >= self.confirm_frames:
                recent = list(buf)[-self.confirm_frames:]
                out[token] = round(sum(recent) / len(recent), 3)
            else:
                out[token] = 0.0
        return out

    # ── helpers ──────────────────────────────────────────────────────────────

    def _normalize(self, lm):
        wx, wy = lm[0].x, lm[0].y
        d = math.dist((lm[0].x, lm[0].y), (lm[9].x, lm[9].y)) + 1e-6
        return [((p.x - wx) / d, (p.y - wy) / d) for p in lm]

    def _finger_states(self, norm):
        # thumb: x-axis
        thumb = norm[4][0] > norm[3][0]
        # other fingers: tip y < pip y (extended upward in normalised space)
        tips = [8, 12, 16, 20]
        pips = [6, 10, 14, 18]
        others = [norm[t][1] < norm[p][1] for t, p in zip(tips, pips)]
        return [thumb] + others  # [thumb, index, middle, ring, pinky]

    def _hand_size(self, lm, h, w):
        """Return the bounding box area of the hand as a fraction of the frame."""
        xs = [p.x for p in lm]
        ys = [p.y for p in lm]
        box_w = (max(xs) - min(xs))
        box_h = (max(ys) - min(ys))
        return box_w * box_h  # fraction of frame (0-1 range)

    def _is_still(self):
        """Check if the hand has been relatively still in recent frames."""
        vels = list(self.velocity_history)
        if len(vels) < 6:
            return True
        recent = vels[-6:]
        avg_speed = sum(math.sqrt(vx**2 + vy**2) for vx, vy in recent) / len(recent)
        return avg_speed < 4.0  # pixels per frame

    def _hand_spread(self, lm):
        """Return the spread of fingers — higher = more open hand."""
        # Distance from index tip to pinky tip, normalized by hand size
        d_index_pinky = math.dist((lm[8].x, lm[8].y), (lm[20].x, lm[20].y))
        d_wrist_middle = math.dist((lm[0].x, lm[0].y), (lm[12].x, lm[12].y)) + 1e-6
        return d_index_pinky / d_wrist_middle

    # ── gesture rules ─────────────────────────────────────────────────────────
    # Each rule returns 0.0 to 0.90.  Higher = more confident.
    # Rules now require STILLNESS for static gestures to prevent mid-motion hallucinations.

    def _rule_stop(self, fingers, still):
        # All 5 fingers extended (open palm facing camera) + MUST be still
        # Also requires thumb extended to distinguish from "wave" etc.
        if fingers[0] and sum(fingers[1:]) >= 4 and still:
            return 0.88
        return 0.0

    def _rule_help(self, fingers, still):
        # Thumbs up: thumb extended, ALL others curled + still
        if fingers[0] and sum(fingers[1:]) == 0 and still:
            return 0.88
        return 0.0

    def _rule_yes(self, fingers):
        # Fist (all curled) + deliberate vertical nodding motion
        if sum(fingers) == 0:
            vels = list(self.velocity_history)
            if len(vels) >= 8:
                vy_vals = [v[1] for v in vels[-8:]]
                changes = sum(
                    1 for i in range(1, len(vy_vals))
                    if vy_vals[i] * vy_vals[i-1] < 0 and abs(vy_vals[i]) > 2.5
                )
                if changes >= 2:  # need at least 2 direction changes (up-down-up)
                    return 0.88
            return 0.35  # fist alone = very low partial (was 0.55)
        return 0.0

    def _rule_no(self, fingers):
        # Index + middle extended, others curled, lateral shaking motion
        if fingers[1] and fingers[2] and not fingers[3] and not fingers[4]:
            vels = list(self.velocity_history)
            if len(vels) >= 8:
                vx_vals = [v[0] for v in vels[-8:]]
                changes = sum(
                    1 for i in range(1, len(vx_vals))
                    if vx_vals[i] * vx_vals[i-1] < 0 and abs(vx_vals[i]) > 2.5
                )
                if changes >= 2:  # need 2+ direction changes
                    return 0.88
            return 0.35  # V-sign alone = very low partial (was 0.55)
        return 0.0

    def _rule_head(self, lm, h, fingers, still):
        # Index pointing up ONLY, tip in upper 30% of frame + still
        tip_y = lm[8].y * h
        if (fingers[1] and not fingers[2] and not fingers[3] and not fingers[4]
                and tip_y < h * 0.30 and still):
            return 0.88
        return 0.0

    def _rule_fever(self, lm, h, fingers, still):
        # Open palm near forehead — wrist in upper 35% + all fingers spread + still
        # Distinguished from STOP by requiring hand to be HIGH
        wrist_y = lm[0].y * h
        if sum(fingers[1:]) >= 3 and wrist_y < h * 0.35 and still:
            return 0.85
        return 0.0

    def _rule_chest(self, lm, h, w, fingers, still):
        # Flat palm at chest height — wrist between 35-55% height
        # Must also be roughly centered horizontally (within middle 70% of frame)
        wrist_y = lm[0].y * h
        wrist_x = lm[0].x * w
        centered = w * 0.15 < wrist_x < w * 0.85
        if (sum(fingers[1:]) >= 3 and h * 0.35 < wrist_y < h * 0.55
                and centered and still):
            return 0.82
        return 0.0

    def _rule_pain(self, lm, h, w, fingers, still):
        # Closed fist in chest/belly zone + STILL (no nodding)
        wrist_y = lm[0].y * h
        wrist_x = lm[0].x * w
        centered = w * 0.15 < wrist_x < w * 0.85
        if (sum(fingers) <= 1 and h * 0.35 < wrist_y < h * 0.70
                and centered and still):
            return 0.82
        return 0.0

    def _rule_stomach(self, lm, h, fingers, still):
        # Flat palm LOW (below 60% height) + still
        wrist_y = lm[0].y * h
        if sum(fingers[1:]) >= 3 and wrist_y > h * 0.62 and still:
            return 0.82
        return 0.0

    def _rule_nausea(self, lm, h, fingers):
        # Hand low + DELIBERATE circular motion (harder threshold)
        wrist_y = lm[0].y * h
        if wrist_y > h * 0.55 and self._is_circular():
            return 0.85
        return 0.0

    def _is_circular(self):
        vels = list(self.velocity_history)
        if len(vels) < 16:  # need more frames to confirm (was 14)
            return False
        total = 0.0
        for i in range(1, len(vels)):
            v1, v2 = vels[i-1], vels[i]
            cross = v1[0]*v2[1] - v1[1]*v2[0]
            dot   = v1[0]*v2[0] + v1[1]*v2[1]
            total += math.atan2(cross, dot + 1e-6)
        return abs(total) > math.pi * 1.2  # need MORE rotation (was 0.8)

    def _rule_water(self, lm, h, fingers, still):
        # W sign: index, middle, ring up. Thumb/pinky off. Near mouth/face.
        wrist_y = lm[0].y * h
        if fingers[1] and fingers[2] and fingers[3] and not fingers[4] and wrist_y < h * 0.45 and still:
            return 0.85
        return 0.0

    def _rule_cough(self, lm, h, w, fingers, still):
        # Fist directly over mouth area
        wrist_y = lm[0].y * h
        wrist_x = lm[0].x * w
        centered = w * 0.25 < wrist_x < w * 0.75
        if sum(fingers) <= 1 and wrist_y < h * 0.40 and centered and still:
            return 0.85
        return 0.0

    def _rule_dizzy(self, lm, h, fingers):
        # Hand circling near the head (high y position)
        wrist_y = lm[0].y * h
        if wrist_y < h * 0.45 and self._is_circular():
            return 0.85
        return 0.0

    def _rule_throat(self, lm, h, w, fingers, still):
        # Index pointing horizontally across neck area
        wrist_y = lm[0].y * h
        wrist_x = lm[0].x * w
        centered = w * 0.25 < wrist_x < w * 0.75
        if fingers[1] and not fingers[2] and not fingers[3] and not fingers[4]:
            if h * 0.35 < wrist_y < h * 0.50 and centered and still:
                return 0.85
        return 0.0

    def _rule_eye(self, lm, h, fingers, still):
        # Index pointing very high near the eye level
        tip_y = lm[8].y * h
        if fingers[1] and not fingers[2] and not fingers[3] and not fingers[4] and tip_y < h * 0.25 and still:
            return 0.85
        return 0.0

    def _rule_thank(self, fingers):
        # Flat palm moving uniformly downward/outward from the chin
        if sum(fingers[1:]) >= 4:
            vels = list(self.velocity_history)
            if len(vels) >= 6:
                vy_vals = [v[1] for v in vels[-6:]]
                if all(vy > 1.5 for vy in vy_vals):
                    return 0.85
        return 0.0

    def _rule_doctor(self, lm, h, fingers, still):
        # V/U sign at chest (index and middle up)
        wrist_y = lm[0].y * h
        if fingers[1] and fingers[2] and not fingers[3] and not fingers[4] and h * 0.35 < wrist_y < h * 0.65 and still:
            return 0.85
        return 0.0

    def _rule_severe(self, fingers):
        # Tight fist shaking fast laterally side-to-side
        if sum(fingers) <= 1:
            vels = list(self.velocity_history)
            if len(vels) >= 8:
                vx_vals = [v[0] for v in vels[-8:]]
                changes = sum(
                    1 for i in range(1, len(vx_vals))
                    if vx_vals[i] * vx_vals[i-1] < 0 and abs(vx_vals[i]) > 3.0
                )
                if changes >= 2:
                    return 0.88
            return 0.35
        return 0.0

    def _rule_medicine(self, lm, h, fingers, still):
        # Pinky extended alone (I sign) held at chest level
        wrist_y = lm[0].y * h
        if not fingers[1] and not fingers[2] and not fingers[3] and fingers[4] and h * 0.35 < wrist_y < h * 0.70 and still:
            return 0.85
        return 0.0

    def _rule_bathroom(self, fingers):
        # Thumb and Pinky extended (Y sign) shaking laterally
        if fingers[0] and not fingers[1] and not fingers[2] and not fingers[3] and fingers[4]:
            vels = list(self.velocity_history)
            if len(vels) >= 8:
                vx_vals = [v[0] for v in vels[-8:]]
                changes = sum(
                    1 for i in range(1, len(vx_vals))
                    if vx_vals[i] * vx_vals[i-1] < 0 and abs(vx_vals[i]) > 2.5
                )
                if changes >= 2:
                    return 0.88
            return 0.40
        return 0.0
