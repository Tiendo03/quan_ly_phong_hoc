import cv2
import numpy as np
from ultralytics import YOLO
import torch
import matplotlib.pyplot as plt
import time
import os

gpio_pins = {
    "PC9" : 73,
    "PC6" : 70,
    "PC8" : 72,
    "PH8" : 232
}
for name, pin in gpio_pins.items():
    if os.path.exists(f"/sys/class/gpio/gpio{pin}"):
        os.system(f"echo {pin} | sudo tee /sys/class/gpio/unexport")
        time.sleep(0.01)
    os.system(f"echo {pin} | sudo tee /sys/class/gpio/export")
    time.sleep(0.1)
    os.system(f"echo out | sudo tee /sys/class/gpio/gpio{pin}/direction")

def tb_on(pin):
    os.system(f'echo 1 | sudo tee /sys/class/gpio/gpio{pin}/value')
def tb_off(pin):
    os.system(f'echo 0 | sudo tee /sys/class/gpio/gpio{pin}/value')
# Thiết lập CUDA nếu có sẵn
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Tải mô hình YOLOv8 với trackings
model = YOLO("yolov8n.pt").to(device)

# Lấy danh sách class của mô hình
model_class_names = model.names
person_width = 0.3  # Chiều rộng trung bình của một người (m)


# Hàm calib camera
# def calibrate_camera():
# calibrationDir = r"/home/orangepi/images"
# imgPathList = glob.glob(os.path.join(calibrationDir, "*.*"))
# nRows, nCols = 9, 6
# termCriteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# worldPtsCur = np.zeros((nRows * nCols, 3), np.float32)
# worldPtsCur[:, :2] = np.mgrid[0:nRows, 0:nCols].T.reshape(-1, 2)
# worldPtsList, imgPtsList = [], []

# for curImgPath in imgPathList:
#     img = cv2.imread(curImgPath)
#     imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     cornersFound, cornersOrg = cv2.findChessboardCorners(
#         imgGray, (nRows, nCols), None
#     )
#     if cornersFound:
#         worldPtsList.append(worldPtsCur)
#         cornersRefined = cv2.cornerSubPix(
#             imgGray, cornersOrg, (11, 11), (-1, -1), termCriteria
#         )
#         imgPtsList.append(cornersRefined)

# repError, camMatrix, distCoeff, _, _ = cv2.calibrateCamera(
#     worldPtsList, imgPtsList, imgGray.shape[::-1], None, None
# )
# return camMatrix, distCoeff


focal_length = 608

# Mở camera/video
cap = cv2.VideoCapture(
    r"/home/orangepi/output_h264.mp4"
)

# plt.ion()
# fig, ax = plt.subplots()
# ax.set_xlim(-10, 10)
# ax.set_ylim(0, 15)
# ax.set_xlabel("Cross-Range Distance (m)")
# ax.set_ylabel("Down-Range Distance (m)")
# ax.set_title("2D Object Tracking Map")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Loi khong doc duoc frame")
        break
    cv2.line(frame, (540, 0), (540, 450), (0, 255, 0), 2)
    cv2.line(frame, (0, 450), (1080, 450), (0, 255, 0), 2)
    frame_resized = cv2.resize(frame, (640, 480))
    results = model.track(frame, conf=0.1, persist=True, tracker="bytetrack.yaml")
    tracked_positions = []

    for r in results:
        boxes = r.boxes
        if boxes is None:
            continue
        for box in boxes:
            cls = int(box.cls[0])
            if model_class_names[cls] == "person":
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                track_id = int(box.id[0]) if box.id is not None else -1
                box_width = x2 - x1

                if box_width > 0.5:
                    distance_bf = (person_width * focal_length) / box_width
                    correction_factor = (
                        1.2
                        if distance_bf < 1.5
                        else 0.9 if distance_bf > 2 else 1.0
                        )
                    distance = (person_width * focal_length) / (
                        box_width * correction_factor
                        )
                    x = (x1 + x2) / 2
                    X_world = (x - frame.shape[1] / 2) * distance / focal_length
                    Z = distance
                    tracked_positions.append((X_world, Z, track_id, distance))

                        # Vẽ bounding box + hiển thị ID và khoảng cách
                    cv2.rectangle(
                            frame,
                            (int(x1), int(y1)),
                            (int(x2), int(y2)),
                            (255, 0, 255),
                            2,
                    )
                    cv2.putText(
                            frame,
                            f"ID: {track_id}",
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
    cv2.imshow("Distance Measurement", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break                

        # Cập nhật bản đồ tọa độ
    # ax.clear()
    # ax.set_xlim(-10, 10)
    # ax.set_ylim(0, 15)
    # ax.set_xlabel("Cross-Range Distance (m)")
    # ax.set_ylabel("Down-Range Distance (m)")
    # ax.set_title("2D Object Tracking Map")

    # ax.scatter(0, 0, marker="^", color="black", label="Camera")
    # for x, y, obj_id, depth in tracked_positions:
    #     ax.scatter(x, y, label=f"ID: {obj_id} ({depth:.1f}m)")
    # ax.legend()
    # plt.draw()
    # plt.pause(0.001)

        # Hiển thị video
    

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()