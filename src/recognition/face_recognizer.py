import numpy as np

class FaceRecognizer:
    """
    Encoder-less face recognition using MediaPipe landmarks.
    Calculates relative distances between key landmarks as a 'signature'.
    """
    def __init__(self):
        # Selected landmarks for robust recognition:
        # Nose tip, eye corners, mouth corners, jawline points
        self.FEATURE_INDICES = [
            1, 4, 152, 33, 263, 61, 291, # Nose, Chin, Eye Corners, Mouth Corners
            10, 151, 9, 8, 168, 6, 197, 195, 5, 4, # Midline
            362, 385, 387, 263, 373, 380, # Left Eye
            33, 160, 158, 133, 153, 144, # Right Eye
            61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291 # Mouth
        ]

    def get_pose_score(self, landmarks):
        """
        Calculates a 'front-facing' score (1.0 is perfect, 0.0 is profile).
        Based on horizontal symmetry between eye outer corners and nose.
        """
        if landmarks is None:
            return 0.0
            
        left_dist = np.linalg.norm(landmarks[33] - landmarks[1]) # Left eye corner to nose
        right_dist = np.linalg.norm(landmarks[263] - landmarks[1]) # Right eye corner to nose
        
        # Symmetry ratio
        if left_dist + right_dist == 0: return 0.0
        ratio = min(left_dist, right_dist) / max(left_dist, right_dist)
        
        return ratio

    def get_face_signature(self, landmarks):
        """
        Generates a normalized feature vector from 3D landmarks.
        """
        if landmarks is None:
            return None
            
        # 1. Select key features
        features = landmarks[self.FEATURE_INDICES].copy()
        
        # 2. Normalize by translating nose tip (index 1) to (0,0,0)
        nose_tip = landmarks[1]
        features = features - nose_tip
        
        # 3. Scale by face width (distance between eye outer corners)
        face_width = np.linalg.norm(landmarks[33] - landmarks[263])
        if face_width > 0:
            features = features / face_width
            
        # 4. Flatten to vector
        return features.flatten().tolist()

    def compare(self, signature1, signature2):
        """
        Calculates similarity between two signatures using Euclidean distance.
        Lower is better (closer match).
        """
        v1 = np.array(signature1)
        v2 = np.array(signature2)
        
        # Euclidean distance
        dist = np.linalg.norm(v1 - v2)
        
        # Normalize to a 0-1 similarity score
        # Calibrated: dist < 0.6 is a strong match, > 1.2 is a clear mismatch
        similarity = np.clip(1.3 - dist, 0, 1)
        return similarity
