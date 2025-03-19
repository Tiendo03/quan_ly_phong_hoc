import cv2
import numpy as np
from ultralytics import YOLO
import torch
from yolox.tracker.byte_tracker import BYTETracker
from types import SimpleNamespace
import matplotlib.pyplot as plt
import glob
import os

# Thiết lập CUDA nếu có sẵn
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Tải mô hình YOLOv8s
model = YOLO("yolov8s.pt").to(device)
model_class_names = model.names

# Thiết lập ByteTrack
tracker_args = SimpleNamespace(
    track_thresh=0.5,
    track_buffer=30,
    match_thresh=0.8,
    aspect_ratio_thresh=1.6,
    min_box_area=10,
    mot20=False,  # Tránh lỗi thiếu biến
)
tracker = BYTETracker(tracker_args)


def calibrate_camera():
    calibrationDir = r"C:\Users\Admin\OneDrive\Documents\A\NCKH-2024\images"
    imgPathList = glob.glob(os.path.join(calibrationDir, "*.*"))
    nRows, nCols = 9, 6
    termCriteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    worldPtsCur = np.zeros((nRows * nCols, 3), np.float32)
    worldPtsCur[:, :2] = np.mgrid[0:nRows, 0:nCols].T.reshape(-1, 2)
    worldPtsList, imgPtsList = [], []

    for curImgPath in imgPathList:
        img = cv2.imread(curImgPath)
        imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cornersFound, cornersOrg = cv2.findChessboardCorners(
            imgGray, (nRows, nCols), None
        )
        if cornersFound:
            worldPtsList.append(worldPtsCur)
            cornersRefined = cv2.cornerSubPix(
                imgGray, cornersOrg, (11, 11), (-1, -1), termCriteria
            )
            imgPtsList.append(cornersRefined)

    repError, camMatrix, distCoeff, _, _ = cv2.calibrateCamera(
        worldPtsList, imgPtsList, imgGray.shape[::-1], None, None
    )
    return camMatrix, distCoeff


focal_length = calibrate_camera()[0][0][0]
print(focal_length)
# Tải mô hình YOLOv8s
model = YOLO("yolov8s.pt").to(device)
model_class_names = model.names
person_width = 0.3  # Chiều rộng trung bình của một người (m)

# Khởi tạo matplotlib
plt.ion()
fig, ax = plt.subplots()
ax.set_xlim(-10, 10)
ax.set_ylim(0, 15)
ax.set_xlabel("Cross-Range Distance (m)")
ax.set_ylabel("Down-Range Distance (m)")
ax.set_title("2D Object Tracking Map")

# Biến lưu vị trí đối tượng
tracked_positions = []

# Mở camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    detections = []
    for r in results:
        boxes = r.boxes
        if boxes is None:
            continue
        for box in boxes:
            cls = int(box.cls[0])
            if model_class_names[cls] == "person":
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                score = box.conf[0].cpu().numpy()
                detections.append([x1, y1, x2, y2, score])

    # Cập nhật tracker
    img_info = (frame.shape[0], frame.shape[1])  # (height, width)
    img_size = frame.shape[:2]  # (height, width)
    if len(detections) > 0:
        detections = np.array(detections, dtype=np.float32)  # Chuyển về NumPy array
        tracked_objects = tracker.update(detections, img_info, img_size)
    else:
        tracked_objects = []

    tracked_positions.clear()  # Xóa dữ liệu cũ
    for obj in tracked_objects:
        x1, y1, x2, y2 = obj.tlbr.tolist()  # Lấy tọa độ từ ByteTrack
        obj_id = obj.track_id  # Object ID
        box_width = x2 - x1
        if box_width > 0.5:
            distance_bf = (person_width * focal_length) / box_width
            if distance_bf < 1.5:
                correction_factor = 1.2
            elif distance_bf > 2:
                correction_factor = 0.9
            else:
                correction_factor = 1.0
            # correction_factor = 1.2 if distance_bf < 1.5 else 1.0  # Bù sai số khi gần
            distance = (person_width * focal_length) / (box_width * correction_factor)
            x = (x1 + x2) / 2
            y = (y1 + y2) / 2
            # Tọa độ thực tế
            X_world = (x - frame.shape[1] / 2) * distance / focal_length
            Z = distance  # Khoảng cách là trục Z
            tracked_positions.append((X_world, Z, obj_id, distance))
            # Vẽ bounding box, hiển thị ID và khoảng cách
            cv2.rectangle(
                frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 255), 2
            )
            cv2.putText(
                frame,
                f"ID: {int(obj_id)}",
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                frame,
                f"Distance: {distance:.2f}m",
                (int(x1), int(y1) - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )
    # Cập nhật bản đồ tọa độ
    ax.clear()
    ax.set_xlim(-10, 10)
    ax.set_ylim(0, 15)
    ax.set_xlabel("Cross-Range Distance (m)")
    ax.set_ylabel("Down-Range Distance (m)")
    ax.set_title("2D Object Tracking Map")

    ax.scatter(0, 0, marker="^", color="black", label="Camera")
    for x, y, obj_id, depth in tracked_positions:
        ax.scatter(x, y, label=f"ID: {obj_id} ({depth:.1f}m)")
    ax.legend()
    plt.draw()
    plt.pause(0.01)
    # Hiển thị hình ảnh
    cv2.imshow("Distance Measurement", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv2.destroyAllWindows()
