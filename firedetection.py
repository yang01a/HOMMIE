import cv2
import numpy as np
import rospy
from std_msgs.msg import String
from cv_bridge import CvBridge
import tkinter as tk
import threading
import time
import os
from tkinter import filedialog

class FireDetectionGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Fire Detection with Labeling")
        self.root.geometry("400x400")

        rospy.init_node('fire_detector', anonymous=True)
        self.fire_pub = rospy.Publisher('/fire_alert', String, queue_size=10)
        self.bridge = CvBridge()

        self.cap = None
        self.is_running = False
        self.sensitivity = 5000
        self.threshold_scale = None
        self.stream_thread = None
        self.previous_frame = None
        self.previous_points = None
        self.last_detection_time = None

        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2()

        self.create_widgets()

        # For storing bounding box coordinates and labeling information
        self.bboxes = []
        self.label = None

    def create_widgets(self):
        self.threshold_scale = tk.Scale(self.root, from_=1000, to_=10000, orient=tk.HORIZONTAL, label="Sensitivity", command=self.update_sensitivity)
        self.threshold_scale.set(self.sensitivity)
        self.threshold_scale.pack(pady=20)

        self.start_button = tk.Button(self.root, text="Start Fire Detection", command=self.toggle_detection)
        self.start_button.pack(pady=10)

        self.status_label = tk.Label(self.root, text="Status: Stopped", fg="red")
        self.status_label.pack(pady=10)

        self.save_button = tk.Button(self.root, text="Save Labels", command=self.save_labels)
        self.save_button.pack(pady=10)

    def update_sensitivity(self, value):
        self.sensitivity = int(value)
        print(f"Updated sensitivity: {self.sensitivity}")

    def toggle_detection(self):
        if self.is_running:
            self.is_running = False
            self.start_button.config(text="Start Fire Detection")
            self.status_label.config(text="Status: Stopped", fg="red")
            self.stop_detection()
        else:
            self.is_running = True
            self.start_button.config(text="Stop Fire Detection")
            self.status_label.config(text="Status: Running", fg="green")
            self.start_detection()

    def start_detection(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Failed to open camera.")
            return
        self.stream_thread = threading.Thread(target=self.detect_fire, daemon=True)
        self.stream_thread.start()

    def stop_detection(self):
        if self.cap:
            self.cap.release()
            self.cap = None
        cv2.destroyAllWindows()

    def save_labels(self):
        if not self.bboxes:
            print("No bounding boxes to save.")
            return

        file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
        if file_path:
            with open(file_path, 'w') as f:
                for bbox in self.bboxes:
                    label_str = f"{self.label} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n"
                    f.write(label_str)
            print(f"Labels saved to {file_path}")

    def detect_fire(self):
        while self.is_running:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, (640, 480))
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            lower_red1 = np.array([0, 120, 120])
            upper_red1 = np.array([10, 255, 255])
            lower_red2 = np.array([160, 120, 120])
            upper_red2 = np.array([180, 255, 255])
            lower_orange = np.array([10, 100, 200])
            upper_orange = np.array([25, 255, 255])
            lower_yellow = np.array([25, 100, 200])
            upper_yellow = np.array([35, 255, 255])

            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            mask3 = cv2.inRange(hsv, lower_orange, upper_orange)
            mask4 = cv2.inRange(hsv, lower_yellow, upper_yellow)
            mask = mask1 | mask2 | mask3 | mask4

            fg_mask = self.bg_subtractor.apply(frame)
            fg_mask = cv2.dilate(fg_mask, None, iterations=2)

            combined_mask = cv2.bitwise_and(mask, fg_mask)

            contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > self.sensitivity:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = float(w) / h
                    if 0.5 < aspect_ratio < 2.0:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        rospy.loginfo("Fire detected!")
                        self.fire_pub.publish("Fire detected!")

                        # Store bounding box coordinates
                        self.bboxes.append((x, y, w, h))

            if self.previous_frame is not None:
                prev_gray = cv2.cvtColor(self.previous_frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                if self.previous_points is None:
                    self.previous_points = cv2.goodFeaturesToTrack(prev_gray, mask=None, **{
                        'maxCorners': 100,
                        'qualityLevel': 0.3,
                        'minDistance': 7,
                        'blockSize': 7
                    })

                next_points, status, error = cv2.calcOpticalFlowPyrLK(prev_gray, gray, self.previous_points, None)

                if next_points is not None:
                    for i, (new, old) in enumerate(zip(next_points, self.previous_points)):
                        a, b = new.ravel()
                        c, d = old.ravel()
                        frame = cv2.line(frame, (a, b), (c, d), (0, 255, 0), 2)

                    rospy.loginfo("Tracking fire movement...")
                    self.previous_points = next_points

            self.previous_frame = frame

            if self.last_detection_time is None or time.time() - self.last_detection_time > 5:
                self.last_detection_time = time.time()
                rospy.loginfo("Fire confirmed!")
                self.fire_pub.publish("Fire confirmed!")

            cv2.imshow("Fire Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.stop_detection()

if __name__ == "__main__":
    root = tk.Tk()
    gui = FireDetectionGUI(root)
    root.mainloop()