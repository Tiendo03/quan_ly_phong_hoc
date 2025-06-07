import cv2
import numpy as np
from ultralytics import YOLO
import torch
import glob
import time
import os

gpio_pins = {"PC9": 73, "PC6": 70, "PC8": 72, "PH8": 232}
for name, pin in gpio_pins.items():
    if os.path.exists(f"/sys/class/gpio/gpio{pin}"):
        os.system(f"echo {pin} | sudo tee /sys/class/gpio/unexport")
        time.sleep(0.01)
    os.system(f"echo {pin} | sudo tee /sys/class/gpio/export")
    time.sleep(0.1)
    os.system(f"echo out | sudo tee /sys/class/gpio/gpio{pin}/direction")

import paho.mqtt.client as mqtt

mqtt_broker = "a7c22cb01920477cb932fb2c8a413336.s1.eu.hivemq.cloud"
mqtt_port = 8883
mqtt_topic = "nckh/camera"
mqtt_username = "doantotnghiep"  # thay bằng username bạn tạo trong HiveMQ Cloud
mqtt_password = "Doantotnghiep2025"  # mật khẩu tương ứng

client = mqtt.Client()
client.username_pw_set(mqtt_username, mqtt_password)

client.tls_set()  # Bắt buộc cho TLS (port 8883)
client.connect(mqtt_broker, mqtt_port, 60)


import json


def send_mqtt(data):
    json_data = json.dumps(data)
    client.publish(mqtt_topic, json_data)
    print(f"Đã gửi MQTT: {json_data}")


def tb_on(pin):
    os.system(f"echo 0 | sudo tee /sys/class/gpio/gpio{pin}/value")


def tb_off(pin):
    os.system(f"echo 1 | sudo tee /sys/class/gpio/gpio{pin}/value")


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

model = YOLO("yolov8n.pt").to(device)
model_class_names = model.names
person_width = 0.3  # Chiều rộng trung bình của một người (m)


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


print(calibrate_camera()[0])
focal_length = calibrate_camera()[0][0][0]

tb_off(73)
tb_off(70)
tb_off(72)
tb_off(232)
mode = "initial"
interval = 5
last_check_time = time.time()
zone_states = {
    1: {"pin": 73, "counter": 0, "active": False},
    2: {"pin": 70, "counter": 0, "active": False},
    3: {"pin": 72, "counter": 0, "active": False},
    4: {"pin": 232, "counter": 0, "active": False},
}
while True:
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    current_time = time.time()
    if current_time - last_check_time < interval:
        continue

    last_check_time = current_time
    print(f"[{time.strftime('%X')}] >>> Đang kiểm tra camera - Chế độ: {mode}")

    cap = cv2.VideoCapture(0)
    time.sleep(0.5)  # cho camera ổn định
    ret, frame = cap.read()

    if not ret:
        print("Lỗi không đọc được frame")
        cap.release()
        continue

    frame_resized = cv2.resize(frame, (640, 480))
    results = model.track(
        frame_resized, conf=0.1, persist=True, tracker="bytetrack.yaml"
    )
    tracked_positions = []
    detected_zones = {1: False, 2: False, 3: False, 4: False}
    for r in results:
        boxes = r.boxes
        if boxes is None:
            continue
        for box in boxes:
            cls = int(box.cls[0])
            if model_class_names[cls] == "person":
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                box_width = x2 - x1
                if box_width > 50:
                    print(">> Có người.")
                    mode = "found"
                    display_duration = 10
                    display = time.time()
                    # no_person_counter = 0
                    while time.time() - display < display_duration:
                        dem_tb = 0
                        latest_data = {}
                        ret, frame = cap.read()
                        if not ret:
                            print("Loi khong doc duoc frame")
                            break
                        results = model.track(
                            frame, conf=0.1, persist=True, tracker="bytetrack.yaml"
                        )
                        tracked_positions = []

                        for r in results:
                            boxes = r.boxes
                            if boxes is None:
                                continue
                            for box in boxes:
                                cls = int(box.cls[0])
                                if model_class_names[cls] != "person":
                                    continue
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                track_id = int(box.id[0]) if box.id is not None else -1
                                box_width = x2 - x1
                                print(f"Box width: {box_width}")
                                distance = (person_width * focal_length) / box_width
                                # print(f"Before: {distance_bf}")
                                # correction_factor = (
                                #     distance_bf + 1.2
                                #     if distance_bf < 1.5
                                #     else 0.6 if distance_bf > 2 else 0.8
                                # )
                                # distance = (person_width * focal_length) / (
                                #     box_width * correction_factor
                                # )
                                # print(f"After: {distance}")
                                x = (x1 + x2) / 2
                                y = (y1 + y2) / 2
                                h, w = frame.shape[:2]
                                tracked_positions.append((track_id, distance))
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
                                if x < w / 2 and y < h / 2 and distance > 2:
                                    print("Nguoi o vung 1")
                                    detected_zones[1] = True
                                    zone = 1
                                    dem_tb += 1
                                    # tb_on(73)
                                else:
                                    print("Khong co nguoi vung 1")
                                if x >= w / 2 and y < h / 2 and distance > 2:
                                    print("Nguoi o vung 2")
                                    detected_zones[2] = True
                                    zone = 2
                                    dem_tb += 1
                                    # tb_on(70)
                                else:
                                    print("Khong co nguoi vung 2")
                                if x < w / 2 and y >= h / 2 and distance > 1:
                                    print("Nguoi o vung 3")
                                    detected_zones[3] = True
                                    zone = 3
                                    dem_tb += 1
                                    # tb_on(72)
                                else:
                                    print("Khong co nguoi vung 3")
                                if x >= w / 2 and y >= h / 2 and distance > 1:
                                    print("Nguoi o vung 4")
                                    detected_zones[4] = True
                                    zone = 4
                                    dem_tb += 1
                                    # tb_on(232)
                                else:
                                    print("Khong co nguoi vung 4")
                                latest_data[track_id] = {
                                    "phong": "1305 - A1",
                                    "so_thiet_bi": dem_tb,
                                    "id": track_id,
                                    "distance": round(distance, 2),
                                    "zone": zone,
                                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                }
                        cv2.imshow("Frame", frame)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            cap.release()
                            cv2.destroyAllWindows()
                            exit()
                    for data in latest_data.values():
                        send_mqtt(data)
                    latest_data.clear()

                    # Vẽ và hiển thị
                    cap.release()
                    cv2.destroyAllWindows()
                    last_check_time = time.time()
                    break
                    # Vùng
        for zone_id, state in zone_states.items():
            if detected_zones[zone_id]:
                state["counter"] = 0
                if not state["active"]:
                    tb_on(state["pin"])
                    state["active"] = True
                    print(f">> Bật vùng {zone_id}")
            else:
                state["counter"] += 1
                print(f"Vùng {zone_id} không có người ({state['counter']}/3 lần)")
                if state["counter"] >= 3:
                    state["counter"] = 2
                    tb_off(state["pin"])
                    state["active"] = False
                    print(f">> Tắt vùng {zone_id} sau 3 lần không phát hiện")

cap.release()
cv2.destroyAllWindows()
