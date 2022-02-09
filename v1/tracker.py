import cv2
import numpy as np
import mediapipe as mp
import yaml


class tracker:
    def __init__(self):
        self.loadSystems()
        self.status = True
        while self.status:
            self.run_camera()

    def loadSystems(self):
        self.pose = mp.solutions.holistic
        self.pose_tracker = self.pose.Holistic()
        self.drawer = mp.solutions.drawing_utils
        self.camera = cv2.VideoCapture(0)

        self.settings = yaml.safe_load(open('config.yaml', 'r'))

    def run_camera(self):
        _, self.frame = self.camera.read()
        self.frame = cv2.flip(self.frame, 1)
        self.pose_est = self.pose_tracker.process(cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB))
        self.frame = cv2.blur(self.frame, (4, 4))
        self.drawer.draw_landmarks(self.frame, self.pose_est.pose_landmarks, self.pose.POSE_CONNECTIONS,
                                   connection_drawing_spec=self.drawer.DrawingSpec(color=(255, 255, 255), thickness=6),
                                   landmark_drawing_spec=self.drawer.DrawingSpec(color=(0, 0, 255), circle_radius=3,
                                                                                 thickness=3))
        cv2.imshow("Window", self.frame)

        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            self.camera.release()
            self.status = False


if __name__ == '__main__':
    tracker()



