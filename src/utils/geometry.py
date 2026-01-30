import numpy as np
import cv2

def calculate_ear(landmarks, left_eye_indices, right_eye_indices):
    """
    Calculates the Eye Aspect Ratio (EAR) for both eyes.
    """
    def eye_ear(eye_landmarks):
        # Vertical distances
        v1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
        v2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
        # Horizontal distance
        h = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
        return (v1 + v2) / (2.0 * h)

    left_ear = eye_ear(landmarks[left_eye_indices])
    right_ear = eye_ear(landmarks[right_eye_indices])
    
    return (left_ear + right_ear) / 2.0

def estimate_head_pose(landmarks, frame_shape):
    """
    Estimates head pose (pitch, yaw, roll) using solvePnP.
    """
    h, w = frame_shape[:2]
    
    # Standard 3D model points (arbitrary but representative)
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip
        (0.0, -330.0, -65.0),        # Chin
        (-225.0, 170.0, -135.0),     # Left eye left corner
        (225.0, 170.0, -135.0),      # Right eye right corner
        (-150.0, -150.0, -125.0),    # Left Mouth corner
        (150.0, -150.0, -125.0)      # Right mouth corner
    ], dtype=np.float32)

    # Corresponding 2D landmark indices for MediaPipe
    # 1: Nose, 152: Chin, 33: Left Eye Left, 263: Right Eye Right, 61: Left Mouth, 291: Right Mouth
    image_points = np.array([
        landmarks[1][:2],
        landmarks[152][:2],
        landmarks[33][:2],
        landmarks[263][:2],
        landmarks[61][:2],
        landmarks[291][:2]
    ], dtype=np.float32)

    focal_length = w
    center = (w/2, h/2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float32)

    dist_coeffs = np.zeros((4,1))
    
    (success, rotation_vector, translation_vector) = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )

    # Rotation matrix
    rmat, _ = cv2.Rodrigues(rotation_vector)
    
    # Euler angles
    _, _, _, _, _, _, angles = cv2.decomposeProjectionMatrix(np.hstack((rmat, translation_vector)))
    pitch, yaw, roll = angles.flatten()

    return pitch, yaw, roll
