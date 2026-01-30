import random
import time
from src.utils.geometry import calculate_ear, estimate_head_pose

class ActiveLiveness:
    """
    Manages active liveness challenges: Blink, Turn Left, Turn Right, Look Up, Look Down.
    """
    def __init__(self):
        self.challenges = ["blink", "look_left", "look_right", "look_up", "look_down"]
        self.current_challenge = None
        self.challenge_start_time = 0
        self.challenge_duration = 7.0 # Increased from 5.0
        self.status = "IDLE" # IDLE, ACTIVE, SUCCESS, FAIL
        
        # Thresholds (Lowered for better UX)
        self.EAR_THRESHOLD = 0.18
        self.YAW_THRESHOLD = 20
        self.PITCH_THRESHOLD = 15
        
        # Landmark indices for MediaPipe (Specific to eyes)
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380] # Corrected indices

    def start_new_challenge(self, initial_landmarks=None, frame_shape=None):
        self.current_challenge = random.choice(self.challenges)
        self.challenge_start_time = time.time()
        self.status = "ACTIVE"
        
        if initial_landmarks is not None and frame_shape is not None:
            p, y, r = estimate_head_pose(initial_landmarks, frame_shape)
            self.initial_pose = (p, y, r)
        else:
            self.initial_pose = (0, 0, 0)
            
        return self.current_challenge

    def verify(self, landmarks, frame_shape):
        if self.status != "ACTIVE":
            return self.status, (0, 0)
        
        if time.time() - self.challenge_start_time > self.challenge_duration:
            self.status = "FAIL"
            return self.status, (0, 0)

        pitch, yaw, roll = estimate_head_pose(landmarks, frame_shape)
        ear = calculate_ear(landmarks, self.LEFT_EYE, self.RIGHT_EYE)
        
        # Calculate deltas from initial pose
        d_pitch = pitch - self.initial_pose[0]
        d_yaw = yaw - self.initial_pose[1]

        passed = False
        if self.current_challenge == "blink":
            if ear < self.EAR_THRESHOLD:
                passed = True
        elif self.current_challenge == "look_left":
            if d_yaw > self.YAW_THRESHOLD:
                passed = True
        elif self.current_challenge == "look_right":
            if d_yaw < -self.YAW_THRESHOLD:
                passed = True
        elif self.current_challenge == "look_up":
            if d_pitch < -self.PITCH_THRESHOLD:
                passed = True
        elif self.current_challenge == "look_down":
            if d_pitch > self.PITCH_THRESHOLD:
                passed = True

        if passed:
            self.status = "SUCCESS"
        
        return self.status, (d_pitch, d_yaw)

if __name__ == "__main__":
    # Test logic
    active = ActiveLiveness()
    print(f"Challenge: {active.start_new_challenge()}")
