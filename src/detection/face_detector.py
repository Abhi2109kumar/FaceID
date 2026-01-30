import cv2
import mediapipe as mp
import numpy as np

class FaceDetector:
    """
    Handles face detection and facial landmark extraction using MediaPipe.
    """
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.drawing_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

    def process(self, frame):
        """
        Processes a single frame and returns landmarks.
        """
        # Convert to RGB as MediaPipe requires it
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        return results

    def get_landmarks(self, results, frame_shape):
        """
        Extracts landmark coordinates as a numpy array.
        """
        if not results.multi_face_landmarks:
            return None
        
        h, w = frame_shape[:2]
        landmarks = []
        for landmark in results.multi_face_landmarks[0].landmark:
            landmarks.append([landmark.x * w, landmark.y * h, landmark.z * w])
            
        return np.array(landmarks)

    def draw_landmarks(self, frame, results):
        """
        Draws landmarks on the frame for visualization.
        """
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.mp_drawing_styles
                    .get_default_face_mesh_tesselation_style()
                )
        return frame

if __name__ == "__main__":
    # Integration test with VideoCapture
    from src.capture.video_stream import VideoCapture
    
    cap = VideoCapture(0).start()
    detector = FaceDetector()
    
    while True:
        grabbed, frame = cap.read()
        if not grabbed:
            break
            
        results = detector.process(frame)
        frame = detector.draw_landmarks(frame, results)
        
        cv2.imshow("Face Detection Test", frame)
        if cv2.waitKey(1) == ord("q"):
            break
            
    cap.stop()
    cv2.destroyAllWindows()
