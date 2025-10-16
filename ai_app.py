from ultralytics import YOLO
import cv2
import cvzone
import time
import threading
import numpy as np

class VideoStream:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src)
        self.ret, self.frame = self.cap.read()
        self.stopped = False
        thread = threading.Thread(target=self.update, args=())
        thread.daemon = True
        thread.start()

    def update(self):
        while not self.stopped:
            self.ret, self.frame = self.cap.read()

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()


# clothing color detection
def get_clothing_color_label(image):
    if image.size == 0:
        return "unknown"
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, w = hsv.shape[:2]
    top_half = hsv[h//12:h//2, w//8:7*w//8]  # Crop top center 3/4 width
    H, S, V = cv2.split(top_half)

    clothing_mask = (S > 50) & (V > 50)
    hue_pixels = H[clothing_mask]
    sat_pixels = S[clothing_mask]
    val_pixels = V[clothing_mask]

    if len(hue_pixels) == 0:
        white_mask = (S < 30) & (V > 200)
        if np.sum(white_mask) / white_mask.size > 0.2:
            return "white"
        black_mask = (V < 50)
        if np.sum(black_mask) / black_mask.size > 0.2:
            return "black"
        return "unknown"

    avg_hue = np.mean(hue_pixels)
    if 0 <= avg_hue <= 10 or 160 <= avg_hue <= 179:
        return "red"
    elif 11 <= avg_hue <= 25:
        return "orange"
    elif 26 <= avg_hue <= 34:
        return "yellow"
    elif 35 <= avg_hue <= 85:
        return "green"
    elif 86 <= avg_hue <= 125:
        return "blue"
    elif 126 <= avg_hue <= 159:
        return "purple"
    return "unknown"


# pose detection
def get_pose_label(keypoints):
    if keypoints is None or len(keypoints) < 17:
        return "unknown"
    nose = keypoints[0]
    l_shoulder, r_shoulder = keypoints[5], keypoints[6]
    l_hip, r_hip = keypoints[11], keypoints[12]
    l_knee, r_knee = keypoints[13], keypoints[14]
    l_ankle, r_ankle = keypoints[15], keypoints[16]

    def distance(a, b):
        return np.linalg.norm(np.array(a) - np.array(b))

    hip_mid = ((l_hip[0]+r_hip[0])/2, (l_hip[1]+r_hip[1])/2)
    knee_mid = ((l_knee[0]+r_knee[0])/2, (l_knee[1]+r_knee[1])/2)
    ankle_mid = ((l_ankle[0]+r_ankle[0])/2, (l_ankle[1]+r_ankle[1])/2)
    shoulder_mid = ((l_shoulder[0]+r_shoulder[0])/2, (l_shoulder[1]+r_shoulder[1])/2)

    if (l_knee[1] > l_hip[1] and r_knee[1] > r_hip[1]):
        if abs(l_ankle[1]-r_ankle[1]) < 30:
            return "sitting"
        else:
            return "crouching"
    if (hip_mid[1] < knee_mid[1] < ankle_mid[1]) and (abs(l_ankle[0] - r_ankle[0]) < 80):
        body_angle = np.arctan2(shoulder_mid[1]-hip_mid[1], shoulder_mid[0]-hip_mid[0])
        if abs(body_angle) < 0.18:
            return "standing"
    footsteps = abs(l_ankle[0] - r_ankle[0])
    footheight = abs(l_ankle[1] - r_ankle[1])
    if footsteps > 120 or footheight > 30:
        return "walking"
    l_wrist, r_wrist = keypoints[9], keypoints[10]
    hand_to_hip = min(distance(l_wrist, hip_mid), distance(r_wrist, hip_mid))
    if hand_to_hip < 120:
        return "holding"
    return "unknown"


RTSP_URL = "rtsp://admin:986gaurav@192.168.1.9:554/Streaming/Channels/301"
MOTION_THRESHOLD_AREA = 1500
CONF_THRESHOLD = 0.4

yolo_obj = YOLO("yolo11l.pt")
yolo_pose = YOLO("yolo11l-pose.pt")

motion_detector = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=25, detectShadows=True)
stream = VideoStream(RTSP_URL)
prev_time = 0

while True:
    ret, frame = stream.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1080, 660))

    # Motion detection mask
    fg_mask = motion_detector.apply(frame)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_DILATE, kernel)

    contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    motion_detected = False
    motion_boxes = []

    for c in contours:
        if cv2.contourArea(c) > MOTION_THRESHOLD_AREA:
            motion_detected = True
            x, y, w, h = cv2.boundingRect(c)
            motion_boxes.append((x, y, w, h))
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)

    motion_status_text = "MOTION DETECTED" if motion_detected else "NO MOTION"
    color = (0, 0, 255) if motion_detected else (0, 255, 0)
    cv2.putText(frame, motion_status_text, (20, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    if motion_detected:
        for (x, y, w, h) in motion_boxes:
            roi = frame[y:y+h, x:x+w]
            results = yolo_obj(roi, stream=False, verbose=False)
            for r in results:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    if conf < CONF_THRESHOLD:
                        continue
                    bx, by, bx1, by1 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    bx_full, by_full = x + bx, y + by
                    bx1_full, by1_full = x + bx1, y + by1
                    crop = frame[by_full:by1_full, bx_full:bx1_full]
                    obj_name = yolo_obj.names[cls]
                    label = obj_name

                    if obj_name == "person":
                        h_crop = crop.shape[0]
                        top_half = crop[0:h_crop//2, :]
                        color_top = get_clothing_color_label(top_half)
                        pose_res = yolo_pose(crop, stream=False, verbose=False)
                        pose_label = "unknown"
                        for pr in pose_res:
                            if pr.keypoints is not None and len(pr.keypoints.xy) > 0:
                                pose_label = get_pose_label(pr.keypoints.xy[0].cpu().numpy())
                            else:
                                pose_label = "No pose detected"
                        label = f"{color_top}, {pose_label} {conf:.2f}"

                    elif obj_name in ["car", "truck", "bus", "motorcycle", "bicycle"]:
                        color_vehicle = get_clothing_color_label(crop)
                        label = f"{color_vehicle} {obj_name}"

                    cv2.rectangle(frame, (bx_full, by_full), (bx1_full, by1_full), (255, 0, 255), 2)
                    cvzone.putTextRect(frame, f"{label} {conf:.2f}", (bx_full, by_full - 10), scale=1, thickness=1)

    # FPS calculation
    curr_time = time.time()
    if prev_time != 0:
        fps = 1 / (curr_time - prev_time)
        cv2.putText(frame, f"FPS: {int(fps)}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    prev_time = curr_time

    cv2.imshow("YOLO + Motion + Attributes", frame)
    cv2.imshow("Motion Mask", fg_mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

stream.stop()
cv2.destroyAllWindows()
