import cv2
import numpy as np

class PassiveLiveness:
    """
    Detects spoofing using frequency analysis (FFT) to identify Moire patterns and screen artifacts.
    """
    def __init__(self, threshold=0.1):
        self.threshold = threshold

    def detect(self, face_roi):
        """
        Analyzes the frequency spectrum and texture sharpness of the face ROI.
        Returns a dictionary with the final score and components for debugging.
        """
        if face_roi is None or face_roi.size == 0:
            return {"score": 0.0, "sharpness": 0.0, "fft": 0.0}
            
        # Convert to grayscale
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
        
        # 1. Sharpness Detection (Laplacian Variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        # Adjusted: 50+ is now considered decent sharpness (was 150)
        sharpness_score = np.clip(laplacian_var / 100.0, 0, 1) 
        
        # 2. Frequency Analysis (FFT)
        dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        
        magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]) + 1)
        
        h, w = gray.shape
        cy, cx = h // 2, w // 2
        r = min(h, w) // 4 
        
        mask = np.ones((h, w), np.uint8)
        mask[cy-r:cy+r, cx-r:cx+r] = 0
        
        high_freq_energy = np.mean(magnitude_spectrum[mask == 1])
        
        # Adjusted FFT thresholds: 
        # Based on your camera: RAW-F of 102.6 is a real face.
        # We now allow up to 130 before starting to penalize as a "grid".
        # Screens typically jump to 200-400 RAW-F.
        fft_spoof_score = np.clip((high_freq_energy - 130) / 100.0, 0, 1)
        fft_liveness_score = 1.0 - fft_spoof_score
        
        # Weighted Final Score
        # Sharpness is a good secondary cue but FFT is the primary spoof blocker
        final_score = (sharpness_score * 0.4) + (fft_liveness_score * 0.6)
        
        return {
            "score": np.clip(final_score, 0, 1),
            "sharpness": sharpness_score,
            "fft": fft_liveness_score,
            "raw_fft": high_freq_energy,
            "raw_sharp": laplacian_var
        }

    def extract_face_roi(self, frame, landmarks):
        """
        Extracts face region from frame based on landmarks.
        Tightened to avoid background pixels.
        """
        if landmarks is None:
            return None
            
        # Focus on the inner part of the face (landmarks 1 to 468)
        # We take a slightly smaller crop to avoid hair/ears/background
        x_min = int(np.min(landmarks[:, 0]))
        y_min = int(np.min(landmarks[:, 1]))
        x_max = int(np.max(landmarks[:, 0]))
        y_max = int(np.max(landmarks[:, 1]))
        
        width = x_max - x_min
        height = y_max - y_min
        
        # Inward padding (Crop 10% from edges to focus on skin)
        x_min += int(width * 0.1)
        x_max -= int(width * 0.1)
        y_min += int(height * 0.1)
        y_max -= int(height * 0.1)
        
        h, w = frame.shape[:2]
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(w, x_max)
        y_max = min(h, y_max)
        
        return frame[y_min:y_max, x_min:x_max]

if __name__ == "__main__":
    # Test stub
    passive = PassiveLiveness()
    print("Passive Liveness Initialized")
