import cv2
import threading
import time

class VideoCapture:
    """
    A threaded video capture class to ensure high FPS and minimize latency.
    """
    def __init__(self, src=0, width=640, height=480):
        self.src = src
        self.cap = cv2.VideoCapture(self.src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()

    def start(self):
        if self.started:
            return None
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()
        return self

    def update(self):
        while self.started:
            grabbed, frame = self.cap.read()
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self):
        with self.read_lock:
            frame = self.frame.copy() if self.frame is not None else None
            grabbed = self.grabbed
        return grabbed, frame

    def stop(self):
        self.started = False
        if self.thread.is_alive():
            self.thread.join()
        self.cap.release()

    def __exit__(self, exec_type, exc_value, traceback):
        self.stop()

if __name__ == "__main__":
    # Simple test for the capture module
    video_getter = VideoCapture(0).start()
    while True:
        (grabbed, frame) = video_getter.read()
        if not grabbed:
            break

        cv2.imshow("Video Capture Test", frame)
        if cv2.waitKey(1) == ord("q"):
            video_getter.stop()
            break
    
    cv2.destroyAllWindows()
