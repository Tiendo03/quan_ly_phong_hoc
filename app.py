import sys
from PyQt6 import QtCore, QtWidgets
from giao_dien.dang_nhap import Ui_MainWindow as LoginUI
from giao_dien.main import Ui_MainWindow as MainUI
import json
import paho.mqtt.client as mqtt


class LoginWindow(QtWidgets.QMainWindow, LoginUI):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        # Giả sử nút login trong dang_nhap.py là self.dang_nhap
        self.dang_nhap.clicked.connect(self.check_credentials)

    def check_credentials(self):
        user = self.user_name.text()
        pwd = self.password.text()
        # Thay "admin"/"123" bằng logic của bạn
        if user == "admin" and pwd == "123":
            self.open_main()
        else:
            QtWidgets.QMessageBox.warning(self, "Lỗi", "Sai tài khoản hoặc mật khẩu")

    def open_main(self):
        self.main_win = MainWindow()
        self.main_win.show()
        self.close()  # đóng login window


class MainWindow(QtWidgets.QMainWindow, MainUI):
    MAX_ROWS = 20

    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # Con trỏ row vòng
        self.next_row = 0

        # Thiết lập MQTT như trước
        self.mqtt_client = mqtt.Client()
        self.mqtt_client.username_pw_set("doantotnghiep", "Doantotnghiep2025")
        self.mqtt_client.tls_set()
        self.mqtt_client.on_connect = self.on_mqtt_connect
        self.mqtt_client.on_message = self.on_mqtt_message
        self.mqtt_client.connect(
            "a7c22cb01920477cb932fb2c8a413336.s1.eu.hivemq.cloud", 8883
        )
        self.mqtt_client.loop_start()

    def on_mqtt_connect(self, client, userdata, flags, rc):
        client.subscribe("nckh/camera")

    def on_mqtt_message(self, client, userdata, msg):
        try:
            data = json.loads(msg.payload.decode())
        except:
            return
        QtCore.QMetaObject.invokeMethod(
            self,
            "update_table_circular",
            QtCore.Qt.ConnectionType.QueuedConnection,
            QtCore.Q_ARG(dict, data),
        )

    @QtCore.pyqtSlot(dict)
    def update_table_circular(self, data):
        """
        Ghi data mới vào row self.next_row, rồi self.next_row = (self.next_row+1)%MAX_ROWS
        """
        row = self.next_row
        # Nếu chưa đủ MAX_ROWS, insert thêm 1 row vào cuối
        if self.tableWidget.rowCount() < self.MAX_ROWS:
            self.tableWidget.insertRow(row)

        # Điền dữ liệu (overwrite nếu row đã tồn tại)
        self.tableWidget.setItem(row, 0, QtWidgets.QTableWidgetItem(str(data["id"])))
        self.tableWidget.setItem(
            row, 1, QtWidgets.QTableWidgetItem(f"{data['distance']:.2f}")
        )
        self.tableWidget.setItem(row, 2, QtWidgets.QTableWidgetItem(str(data["zone"])))
        self.tableWidget.setItem(row, 3, QtWidgets.QTableWidgetItem(data["timestamp"]))

        # Cập nhật con trỏ, vòng lại khi chạm MAX_ROWS
        self.next_row = (self.next_row + 1) % self.MAX_ROWS


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    login = LoginWindow()
    login.show()
    sys.exit(app.exec())
