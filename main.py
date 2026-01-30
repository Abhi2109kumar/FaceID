import cv2
import time
import numpy as np
from src.capture.video_stream import VideoCapture
from src.detection.face_detector import FaceDetector
from src.liveness.active.challenge import ActiveLiveness
from src.liveness.passive.texture_analysis import PassiveLiveness
from src.recognition.face_recognizer import FaceRecognizer
from src.utils.database import DatabaseManager

class FaceIDSystem:
    def __init__(self):
        self.cap = VideoCapture(0).start()
        self.detector = FaceDetector()
        self.active_liveness = ActiveLiveness()
        self.passive_liveness = PassiveLiveness()
        self.recognizer = FaceRecognizer()
        self.db = DatabaseManager()
        
        self.mode = "IDLE" # IDLE, REGISTER, LOGIN
        self.session_started = False
        self.active_challenge = None
        self.final_result = "PRESS 'R' TO REGISTER OR 'L' TO LOGIN"
        self.scores = {"passive": [], "active": "PENDING", "pose_deltas": (0,0)}
        self.signature_buffer = [] # (pose_score, signature)
        self.current_user_name = None

    def run(self):
        print("Starting FaceID System...")
        print("Keys: 'r' - Register New User, 'l' - Login, 'q' - Quit")
        
        while True:
            grabbed, frame = self.cap.read()
            if not grabbed:
                break
                
            results = self.detector.process(frame)
            landmarks = self.detector.get_landmarks(results, frame.shape)
            
            if landmarks is not None:
                # 1. Passive Liveness (Continuous)
                face_roi = self.passive_liveness.extract_face_roi(frame, landmarks)
                passive_result = self.passive_liveness.detect(face_roi)
                passive_score = passive_result["score"]
                self.scores["passive"].append(passive_score)
                
                # 2. Collect Signature candidates during session
                if self.session_started:
                    pose_score = self.recognizer.get_pose_score(landmarks)
                    signature = self.recognizer.get_face_signature(landmarks)
                    self.signature_buffer.append((pose_score, signature))
                
                # 3. Workflows (If started)
                if self.session_started:
                    status, deltas = self.active_liveness.verify(landmarks, frame.shape)
                    self.scores["active"] = status
                    self.scores["pose_deltas"] = deltas
                    
                    if status == "SUCCESS":
                        # Liveness Checked, now process based on mode
                        avg_passive = np.mean(self.scores["passive"]) if self.scores["passive"] else 0
                        if avg_passive > 0.4:
                            # Pick the best (most front-facing) signature from the session
                            # This prevents matching against a profile view
                            self.signature_buffer.sort(key=lambda x: x[0], reverse=True)
                            best_sig = self.signature_buffer[0][1] if self.signature_buffer else None
                            
                            if self.mode == "REGISTER":
                                self.db.register_user(self.current_user_name, best_sig)
                                self.final_result = f"REGISTERED: {self.current_user_name}"
                            elif self.mode == "LOGIN":
                                self.match_user(best_sig)
                        else:
                            self.final_result = "SPOOF DETECTED (TEXTURE)"
                        self.session_started = False
                        self.mode = "IDLE"
                    elif status == "FAIL":
                        self.final_result = "SPOOF DETECTED (ACTION)"
                        self.session_started = False
                        self.mode = "IDLE"
                else:
                    self.scores["pose_deltas"] = (0, 0)
                
                # Visualization
                frame = self.detector.draw_landmarks(frame, results)
                self.draw_ui(frame, passive_result)
            
            cv2.imshow("FaceID Liveness System", frame)
            
            key = cv2.waitKey(1)
            if key == ord("q"):
                break
            elif key == ord("r") and not self.session_started:
                # Basic input for name
                self.current_user_name = input("Enter name for registration: ")
                if self.current_user_name:
                    self.mode = "REGISTER"
                    self.start_session(landmarks, frame.shape)
            elif key == ord("l") and not self.session_started:
                self.mode = "LOGIN"
                self.start_session(landmarks, frame.shape)
                
        self.cap.stop()
        cv2.destroyAllWindows()

    def start_session(self, landmarks, frame_shape):
        self.session_started = True
        self.final_result = "AUTHENTICATING..."
        self.active_challenge = self.active_liveness.start_new_challenge(landmarks, frame_shape)
        self.scores["passive"] = []
        self.signature_buffer = [] # Reset buffer
        print(f"Session Started ({self.mode}). Challenge: {self.active_challenge}")

    def match_user(self, current_signature):
        users = self.db.get_all_users()
        best_match = None
        highest_similarity = 0
        
        for name, data in users.items():
            similarity = self.recognizer.compare(current_signature, data["signature"])
            if similarity > highest_similarity:
                highest_similarity = similarity
                best_match = name
        
        # Threshold for recognition: 0.70 similarity
        if highest_similarity > 0.70:
            self.final_result = f"WELCOME BACK, {best_match.upper()}!"
        else:
            self.final_result = "UNKNOWN USER / FACE NOT RECOGNIZED"

    def draw_ui(self, frame, passive_result):
        # Background for text
        cv2.rectangle(frame, (10, 10), (450, 200), (0, 0, 0), -1)
        
        # Display Status
        color = (0, 255, 0) if "VERIFIED" in self.final_result or "WELCOME" in self.final_result or "REGISTERED" in self.final_result else (0, 0, 255)
        if "AUTHENTICATING" in self.final_result: color = (0, 255, 255)
        if "PRESS" in self.final_result: color = (255, 255, 255)
        
        # Split long results into two lines if needed
        cv2.putText(frame, f"STATUS: {self.final_result}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Display Passive Scores (Simplified)
        cv2.putText(frame, f"Liveness Score: {passive_result['score']:.2f}", (20, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(frame, f"Texture: {passive_result['fft']:.2f} Sharpness: {passive_result['sharpness']:.2f}", (20, 95), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # Display Active Challenge or Pose info
        if self.session_started:
            mode_text = f"MODE: {self.mode}"
            cv2.putText(frame, mode_text, (20, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(frame, f"CHALLENGE: {self.active_challenge.upper()}", (20, 145), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Display Pose Deltas during challenge
            deltas = self.scores.get("pose_deltas", (0, 0))
            cv2.putText(frame, f"P-Delta: {deltas[0]:.1f} Y-Delta: {deltas[1]:.1f}", (20, 190), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Timer bar
            elapsed = time.time() - self.active_liveness.challenge_start_time
            remaining = max(0, self.active_liveness.challenge_duration - elapsed)
            width = int((remaining / self.active_liveness.challenge_duration) * 410)
            cv2.rectangle(frame, (20, 160), (20 + width, 170), (0, 255, 255), -1)

if __name__ == "__main__":
    system = FaceIDSystem()
    system.run()
