# gesture_engine.py — FULL REWRITE

import math
from collections import deque


class GestureEngine:
    def __init__(self, buffer_size=24, confirm_frames=18, confirm_threshold=0.74):
        self.buffer_size = buffer_size
        self.confirm_frames = confirm_frames
        self.confirm_threshold = confirm_threshold  # LOWERED from 0.82 → 0.74

        self.tokens = [
            "PAIN", "FEVER", "NAUSEA", "HEAD", "CHEST",
            "STOMACH", "YES", "NO", "HELP", "STOP",
            "WATER", "COUGH", "DIZZY", "THROAT", "EYE",
            "THANK", "DOCTOR", "SEVERE", "MEDICINE", "BATHROOM"
        ]

        self.confidence_buffers = {t: deque(maxlen=buffer_size) for t in self.tokens}
        # Smoothed display confidence (exponential moving average) — avoids UI jitter
        self.smooth_conf = {t: 0.0 for t in self.tokens}
        self.smooth_alpha = 0.25  # EMA factor

        self.velocity_history = deque(maxlen=24)
        self.prev_wrist = None

        # Per-token cooldown — different tokens don't block each other
        self.token_cooldowns = {t: 0 for t in self.tokens}
        self.global_cooldown = 0  # only used right after commit

        # Track last committed token to prevent immediate repeat
        self.last_committed = None
        self.last_committed_timer = 0

    def reset(self):
        """Called after a token is committed."""
        for buf in self.confidence_buffers.values():
            buf.clear()
        for t in self.smooth_conf:
            self.smooth_conf[t] = 0.0
        self.velocity_history.clear()
        self.prev_wrist = None
        self.global_cooldown = 20  # short global pause after commit

    def process(self, hand_landmarks_list, frame_shape):
        # Decrement global cooldown
        if self.global_cooldown > 0:
            self.global_cooldown -= 1
            return None, 0.0, {}

        # Decrement per-token cooldowns
        for t in self.token_cooldowns:
            if self.token_cooldowns[t] > 0:
                self.token_cooldowns[t] -= 1

        if self.last_committed_timer > 0:
            self.last_committed_timer -= 1
        else:
            self.last_committed = None

        if not hand_landmarks_list:
            for buf in self.confidence_buffers.values():
                buf.append(0.0)
            self._update_smooth({t: 0.0 for t in self.tokens})
            self.prev_wrist = None
            return None, 0.0, self._get_smooth_scores()

        h, w = frame_shape[:2]
        hand = hand_landmarks_list[0]
        lm = hand.landmark if hasattr(hand, 'landmark') else hand

        # Reject tiny/noise detections
        hand_size = self._hand_size(lm, h, w)
        if hand_size < 0.04:
            for buf in self.confidence_buffers.values():
                buf.append(0.0)
            self._update_smooth({t: 0.0 for t in self.tokens})
            return None, 0.0, self._get_smooth_scores()

        norm = self._normalize(lm)
        fingers = self._finger_states(norm)
        curl = self._finger_curl_amounts(norm)  # NEW: per-finger curl 0.0-1.0

        # Velocity tracking
        wrist_px = (lm[0].x * w, lm[0].y * h)
        if self.prev_wrist:
            vx = wrist_px[0] - self.prev_wrist[0]
            vy = wrist_px[1] - self.prev_wrist[1]
        else:
            vx, vy = 0.0, 0.0
        self.prev_wrist = wrist_px
        self.velocity_history.append((vx, vy))

        still = self._is_still()
        very_still = self._is_very_still()  # NEW: stricter stillness check

        scores = {
            "STOP":     self._rule_stop(fingers, still),
            "HELP":     self._rule_help(fingers, still),
            "YES":      self._rule_yes(fingers, very_still),
            "NO":       self._rule_no(fingers),
            "HEAD":     self._rule_head(lm, h, w, fingers, still),
            "FEVER":    self._rule_fever(lm, h, fingers, still),
            "CHEST":    self._rule_chest(lm, h, w, fingers, still),
            "PAIN":     self._rule_pain(lm, h, w, fingers, very_still),
            "STOMACH":  self._rule_stomach(lm, h, fingers, still),
            "NAUSEA":   self._rule_nausea(lm, h, fingers),
            "WATER":    self._rule_water(lm, h, w, fingers, still),
            "COUGH":    self._rule_cough(lm, h, w, fingers, still),
            "DIZZY":    self._rule_dizzy(lm, h, fingers),
            "THROAT":   self._rule_throat(lm, h, w, fingers, still),
            "EYE":      self._rule_eye(lm, h, w, fingers, curl, still),
            "THANK":    self._rule_thank(lm, h, fingers),
            "DOCTOR":   self._rule_doctor(lm, h, w, fingers, still),
            "SEVERE":   self._rule_severe(fingers),
            "MEDICINE": self._rule_medicine(lm, h, w, fingers, still),
            "BATHROOM": self._rule_bathroom(fingers),
        }

        # Apply per-token cooldowns
        for t in self.tokens:
            if self.token_cooldowns[t] > 0:
                scores[t] = 0.0
            if t == self.last_committed:
                scores[t] = min(scores[t], 0.3)  # suppress immediate repeat

        # ── MUTUAL EXCLUSION (improved — use spread score, not just position) ──
        # Palm group: winner must beat second-best by margin
        palm_group = ["STOP", "FEVER", "CHEST", "STOMACH"]
        self._suppress_group(scores, palm_group, margin=0.12)

        # Fist group: COUGH is distinguished by HEIGHT (face-level), PAIN by mid-body
        # YES is distinguished by MOTION — but add extra margin to make it clearer
        fist_group = ["PAIN", "YES", "HELP", "COUGH"]
        self._suppress_group(scores, fist_group, margin=0.10)

        # Index group: HEAD vs EYE now have very different rules (see below)
        idx_group = ["HEAD", "EYE", "THROAT"]
        self._suppress_group(scores, idx_group, margin=0.10)

        # V-sign group: DOCTOR vs NO (both use index+middle)
        vsign_group = ["DOCTOR", "NO"]
        self._suppress_group(scores, vsign_group, margin=0.10)

        # Append to buffers
        for token, score in scores.items():
            self.confidence_buffers[token].append(float(score))

        # Update EMA smooth scores
        self._update_smooth(scores)

        # Compute weighted average (more weight on recent frames)
        avg_scores = self._compute_weighted_avg()

        # Find best
        best_token = None
        best_conf = 0.0
        for token, avg in avg_scores.items():
            if avg > best_conf:
                best_conf = avg
                best_token = token

        if best_conf >= self.confirm_threshold:
            return best_token, best_conf, self._get_smooth_scores()
        if best_conf > 0.25:
            return best_token, best_conf, self._get_smooth_scores()
        return None, 0.0, self._get_smooth_scores()

    def mark_committed(self, token):
        """Call this when a token is committed so we can suppress repeat detection."""
        self.last_committed = token
        self.last_committed_timer = 30  # 30 frames suppression
        self.token_cooldowns[token] = 25  # this specific token cools down

    # ── HELPERS ──────────────────────────────────────────────────────────────

    def _suppress_group(self, scores, group, margin=0.10):
        """
        In a competing group, suppress all tokens except the best.
        If the best doesn't beat second-best by `margin`, suppress everyone
        (ambiguous gesture — don't commit anything).
        """
        active = {g: scores[g] for g in group if scores[g] > 0}
        if len(active) <= 1:
            return
        sorted_vals = sorted(active.values(), reverse=True)
        best_val = sorted_vals[0]
        second_val = sorted_vals[1] if len(sorted_vals) > 1 else 0
        best_key = max(active, key=active.get)

        if best_val - second_val < margin:
            # Too ambiguous — suppress all
            for g in group:
                scores[g] = 0.0
        else:
            # Suppress all except winner
            for g in group:
                if g != best_key:
                    scores[g] = 0.0

    def _normalize(self, lm):
        wx, wy = lm[0].x, lm[0].y
        d = math.dist((lm[0].x, lm[0].y), (lm[9].x, lm[9].y)) + 1e-6
        return [((p.x - wx) / d, (p.y - wy) / d) for p in lm]

    def _finger_states(self, norm):
        thumb = norm[4][0] > norm[3][0]
        tips = [8, 12, 16, 20]
        pips = [6, 10, 14, 18]
        others = [norm[t][1] < norm[p][1] for t, p in zip(tips, pips)]
        return [thumb] + others

    def _finger_curl_amounts(self, norm):
        """
        Returns curl amount per finger: 0.0 = fully open, 1.0 = fully curled.
        More nuanced than boolean finger_states.
        """
        tips = [8, 12, 16, 20]
        pips = [6, 10, 14, 18]
        mcps = [5, 9, 13, 17]
        curls = []
        for tip, pip, mcp in zip(tips, pips, mcps):
            # tip_y relative to pip_y in normalised space (higher y = more curled)
            tip_y = norm[tip][1]
            pip_y = norm[pip][1]
            mcp_y = norm[mcp][1]
            range_ = abs(mcp_y - pip_y) + 1e-6
            curl = max(0.0, min(1.0, (tip_y - pip_y) / range_))
            curls.append(curl)
        return curls  # [index, middle, ring, pinky]

    def _hand_size(self, lm, h, w):
        xs = [p.x for p in lm]
        ys = [p.y for p in lm]
        return (max(xs) - min(xs)) * (max(ys) - min(ys))

    def _is_still(self):
        vels = list(self.velocity_history)
        if len(vels) < 6:
            return True
        recent = vels[-6:]
        avg_speed = sum(math.sqrt(vx**2 + vy**2) for vx, vy in recent) / len(recent)
        return avg_speed < 5.0

    def _is_very_still(self):
        """Stricter stillness — used for PAIN to avoid YES false positives."""
        vels = list(self.velocity_history)
        if len(vels) < 8:
            return True
        recent = vels[-8:]
        avg_speed = sum(math.sqrt(vx**2 + vy**2) for vx, vy in recent) / len(recent)
        return avg_speed < 2.5  # tighter than _is_still

    def _is_circular(self):
        vels = list(self.velocity_history)
        if len(vels) < 16:
            return False
        total = 0.0
        for i in range(1, len(vels)):
            v1, v2 = vels[i-1], vels[i]
            cross = v1[0]*v2[1] - v1[1]*v2[0]
            dot = v1[0]*v2[0] + v1[1]*v2[1]
            total += math.atan2(cross, dot + 1e-6)
        return abs(total) > math.pi * 1.4

    def _hand_spread(self, lm):
        d_index_pinky = math.dist((lm[8].x, lm[8].y), (lm[20].x, lm[20].y))
        d_wrist_middle = math.dist((lm[0].x, lm[0].y), (lm[12].x, lm[12].y)) + 1e-6
        return d_index_pinky / d_wrist_middle

    def _update_smooth(self, scores):
        for t in self.tokens:
            s = scores.get(t, 0.0)
            self.smooth_conf[t] = (self.smooth_alpha * s
                                   + (1 - self.smooth_alpha) * self.smooth_conf[t])

    def _get_smooth_scores(self):
        return {t: round(self.smooth_conf[t], 3) for t in self.tokens}

    def _compute_weighted_avg(self):
        """Weighted average: recent frames have more weight (linear ramp)."""
        out = {}
        for token, buf in self.confidence_buffers.items():
            if len(buf) < self.confirm_frames:
                out[token] = 0.0
                continue
            recent = list(buf)[-self.confirm_frames:]
            n = len(recent)
            weights = [i + 1 for i in range(n)]  # 1,2,3,...,n
            total_w = sum(weights)
            out[token] = round(sum(w * v for w, v in zip(weights, recent)) / total_w, 3)
        return out

    # ── GESTURE RULES (ALL REWRITTEN) ────────────────────────────────────────

    def _rule_stop(self, fingers, still):
        # All 5 fingers extended, wide spread, still
        if fingers[0] and sum(fingers[1:]) >= 4 and still:
            return 0.88
        return 0.0

    def _rule_help(self, fingers, still):
        # Thumbs up: thumb up, all others curled, still
        if fingers[0] and sum(fingers[1:]) == 0 and still:
            return 0.88
        return 0.0

    def _rule_yes(self, fingers, very_still):
        # Fist + deliberate vertical nodding
        # very_still check: if hand is very still, it's probably PAIN not YES
        if sum(fingers) == 0:
            vels = list(self.velocity_history)
            if len(vels) >= 8:
                vy_vals = [v[1] for v in vels[-8:]]
                changes = sum(
                    1 for i in range(1, len(vy_vals))
                    if vy_vals[i] * vy_vals[i-1] < 0 and abs(vy_vals[i]) > 3.0
                )
                if changes >= 2:
                    return 0.88
            # Fist alone with very_still = likely PAIN, don't score YES
            if very_still:
                return 0.0
            return 0.25  # reduced from 0.35
        return 0.0

    def _rule_no(self, fingers):
        # V-sign + lateral shake. IMPORTANT: must not be at chest level (that's DOCTOR)
        if fingers[1] and fingers[2] and not fingers[3] and not fingers[4]:
            vels = list(self.velocity_history)
            if len(vels) >= 8:
                vx_vals = [v[0] for v in vels[-8:]]
                changes = sum(
                    1 for i in range(1, len(vx_vals))
                    if vx_vals[i] * vx_vals[i-1] < 0 and abs(vx_vals[i]) > 3.0
                )
                if changes >= 2:
                    return 0.88
            return 0.25
        return 0.0

    def _rule_head(self, lm, h, w, fingers, still):
        # Index only, tip in upper 35% of frame, wrist also HIGH (upper 50%)
        # Distinct from THROAT (which is mid-frame horizontal)
        tip_y = lm[8].y * h
        wrist_y = lm[0].y * h
        wrist_x = lm[0].x * w
        centered = w * 0.20 < wrist_x < w * 0.80
        if (fingers[1] and not fingers[2] and not fingers[3] and not fingers[4]
                and tip_y < h * 0.32 and wrist_y < h * 0.50 and centered and still):
            return 0.88
        return 0.0

    def _rule_fever(self, lm, h, fingers, still):
        # Open palm, wrist in upper 30% (distinctly higher than CHEST which is 35-55%)
        wrist_y = lm[0].y * h
        spread = self._hand_spread(lm)
        if (sum(fingers[1:]) >= 3 and wrist_y < h * 0.30
                and spread > 0.5 and still):
            return 0.88
        return 0.0

    def _rule_chest(self, lm, h, w, fingers, still):
        # Open palm, wrist strictly between 35-55%, centered
        wrist_y = lm[0].y * h
        wrist_x = lm[0].x * w
        centered = w * 0.15 < wrist_x < w * 0.85
        if (sum(fingers[1:]) >= 3 and h * 0.35 < wrist_y < h * 0.55
                and centered and still):
            return 0.85
        return 0.0

    def _rule_pain(self, lm, h, w, fingers, very_still):
        # Closed fist, mid-body zone, VERY still (to separate from YES nodding)
        # wrist must be between 35-70% height
        wrist_y = lm[0].y * h
        wrist_x = lm[0].x * w
        centered = w * 0.15 < wrist_x < w * 0.85
        fist = sum(fingers) <= 1
        if fist and h * 0.35 < wrist_y < h * 0.70 and centered and very_still:
            return 0.88
        # Partial: fist in zone but not very still
        if fist and h * 0.35 < wrist_y < h * 0.70 and centered:
            return 0.40
        return 0.0

    def _rule_stomach(self, lm, h, fingers, still):
        # Open palm LOW (wrist > 62%) + still + SPREAD fingers
        # Added spread check and stricter wrist threshold vs CHEST
        wrist_y = lm[0].y * h
        spread = self._hand_spread(lm)
        if (sum(fingers[1:]) >= 3 and wrist_y > h * 0.65
                and spread > 0.4 and still):
            return 0.85
        return 0.0

    def _rule_nausea(self, lm, h, fingers):
        wrist_y = lm[0].y * h
        if wrist_y > h * 0.55 and self._is_circular():
            return 0.88
        return 0.0

    def _rule_water(self, lm, h, w, fingers, still):
        # W-sign: index, middle, ring up. Near face/mouth.
        wrist_y = lm[0].y * h
        wrist_x = lm[0].x * w
        centered = w * 0.15 < wrist_x < w * 0.85
        if (fingers[1] and fingers[2] and fingers[3]
                and not fingers[0] and not fingers[4]
                and wrist_y < h * 0.45 and centered and still):
            return 0.88
        return 0.0

    def _rule_cough(self, lm, h, w, fingers, still):
        # Fist near FACE (wrist < 38%), centered
        # Tighter height band than PAIN (which is 35-70%) to avoid overlap
        wrist_y = lm[0].y * h
        wrist_x = lm[0].x * w
        centered = w * 0.25 < wrist_x < w * 0.75
        fist = sum(fingers) <= 1
        if fist and wrist_y < h * 0.38 and centered and still:
            return 0.88
        return 0.0

    def _rule_dizzy(self, lm, h, fingers):
        wrist_y = lm[0].y * h
        if wrist_y < h * 0.45 and self._is_circular():
            return 0.88
        return 0.0

    def _rule_throat(self, lm, h, w, fingers, still):
        # Index horizontal, neck zone (40-55% height), centered
        # Distinct from HEAD: wrist is MID not HIGH
        wrist_y = lm[0].y * h
        wrist_x = lm[0].x * w
        centered = w * 0.20 < wrist_x < w * 0.80
        # Check finger is roughly horizontal (x extent > y extent)
        tip_x = lm[8].x * w
        tip_y = lm[8].y * h
        horizontal_extent = abs(tip_x - wrist_x)
        vertical_extent = abs(tip_y - wrist_y)
        is_horizontal = horizontal_extent > vertical_extent * 0.8

        if (fingers[1] and not fingers[2] and not fingers[3] and not fingers[4]
                and h * 0.38 < wrist_y < h * 0.52
                and centered and is_horizontal and still):
            return 0.88
        return 0.0

    def _rule_eye(self, lm, h, w, fingers, curl, still):
        """
        REWRITTEN: Now uses index+thumb pinch near eye level.
        Old rule (tip_y < 25%) was too similar to HEAD (tip_y < 32%).
        New rule: index extended, thumb touching tip (pinch gesture), near face.
        """
        tip_y = lm[8].y * h
        wrist_y = lm[0].y * h
        # Thumb tip to index tip distance (pinch check)
        dx = lm[4].x - lm[8].x
        dy = lm[4].y - lm[8].y
        pinch_dist = math.sqrt(dx**2 + dy**2)

        # Eye sign: index pointing up, thumb tucked close to tip (pinch or near-pinch)
        # wrist must be upper half, tip in upper 30%
        if (fingers[1] and not fingers[2] and not fingers[3] and not fingers[4]
                and pinch_dist < 0.06  # thumb near index tip
                and tip_y < h * 0.30
                and wrist_y < h * 0.45 and still):
            return 0.88
        return 0.0

    def _rule_thank(self, lm, h, fingers):
        # Open palm moving downward/outward from face level
        wrist_y = lm[0].y * h
        if sum(fingers[1:]) >= 4 and wrist_y < h * 0.45:
            vels = list(self.velocity_history)
            if len(vels) >= 6:
                vy_vals = [v[1] for v in vels[-6:]]
                if all(vy > 1.5 for vy in vy_vals):
                    return 0.88
        return 0.0

    def _rule_doctor(self, lm, h, w, fingers, still):
        # V/U sign at CHEST level (index + middle up), NO lateral shake
        # Distinct from NO: no lateral shake required; distinct from WATER: only 2 fingers
        wrist_y = lm[0].y * h
        wrist_x = lm[0].x * w
        centered = w * 0.20 < wrist_x < w * 0.80
        vels = list(self.velocity_history)
        # Check there's no lateral shaking (that would be NO)
        if len(vels) >= 6:
            vx_vals = [abs(v[0]) for v in vels[-6:]]
            avg_lateral = sum(vx_vals) / len(vx_vals)
            if avg_lateral > 3.0:
                return 0.0  # shaking = NO, not DOCTOR
        if (fingers[1] and fingers[2] and not fingers[3] and not fingers[4]
                and h * 0.35 < wrist_y < h * 0.65 and centered and still):
            return 0.88
        return 0.0

    def _rule_severe(self, fingers):
        # Tight fist shaking fast side-to-side
        if sum(fingers) <= 1:
            vels = list(self.velocity_history)
            if len(vels) >= 8:
                vx_vals = [v[0] for v in vels[-8:]]
                changes = sum(
                    1 for i in range(1, len(vx_vals))
                    if vx_vals[i] * vx_vals[i-1] < 0 and abs(vx_vals[i]) > 3.5
                )
                if changes >= 2:
                    return 0.88
            return 0.25
        return 0.0

    def _rule_medicine(self, lm, h, w, fingers, still):
        # Pinky alone extended (ASL "I" sign) at chest level
        wrist_y = lm[0].y * h
        wrist_x = lm[0].x * w
        centered = w * 0.15 < wrist_x < w * 0.85
        if (not fingers[0] and not fingers[1] and not fingers[2]
                and not fingers[3] and fingers[4]
                and h * 0.35 < wrist_y < h * 0.70 and centered and still):
            return 0.88
        return 0.0

    def _rule_bathroom(self, fingers):
        # Thumb + pinky extended (ASL "Y" / shaka) + lateral shake
        if (fingers[0] and not fingers[1] and not fingers[2]
                and not fingers[3] and fingers[4]):
            vels = list(self.velocity_history)
            if len(vels) >= 8:
                vx_vals = [v[0] for v in vels[-8:]]
                changes = sum(
                    1 for i in range(1, len(vx_vals))
                    if vx_vals[i] * vx_vals[i-1] < 0 and abs(vx_vals[i]) > 2.5
                )
                if changes >= 2:
                    return 0.88
            return 0.30
        return 0.0
