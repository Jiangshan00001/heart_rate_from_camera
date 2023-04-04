#coding:utf-8
# 导入所需的库
import sys
import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication, QMainWindow
from PyQt5.QtCore import QTimer
from pyqtgraph import PlotWidget, plot
from scipy.ndimage import gaussian_filter1d


# 导入scipy中的butter和filtfilt函数
from scipy.signal import butter, filtfilt

# 导入time模块，用于计算帧率
import time

# 定义一个窗口类，继承自QMainWindow
class Ui_MainWindow(QMainWindow):
    # 初始化方法
    def __init__(self):
        super().__init__()
        # 设置窗口的标题和大小
        self.setWindowTitle("Python Program")
        self.resize(800, 600)
        # 创建一个视频捕获对象，参数为0表示使用默认的摄像头
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # 创建一个定时器，用于定时更新视频帧和曲线数据
        self.timer = QTimer()
        # 创建一个人脸检测器，使用opencv自带的级联分类器
        self.face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")#)
        # 创建一个空的numpy数组，用于存储原始数据和心率数据
        self.raw_data = np.array([])
        self.filtered_data=np.array([])
        self.heart_rate_data = np.array([0])
        # 创建一个布局对象，用于放置控件
        self.layout = QtWidgets.QVBoxLayout()
        # 创建两个按钮，分别用于开始和结束视频
        self.start_button = QtWidgets.QPushButton("开始视频")
        self.stop_button = QtWidgets.QPushButton("结束视频")
        # 将按钮添加到布局中，并设置按钮的信号和槽函数
        self.layout.addWidget(self.start_button)
        self.layout.addWidget(self.stop_button)
        self.start_button.clicked.connect(self.start_video)
        self.stop_button.clicked.connect(self.stop_video)
        # 创建一个标签，用于显示视频帧
        self.video_label = QtWidgets.QLabel()
        # 将标签添加到布局中，并设置标签的大小和对齐方式
        self.layout.addWidget(self.video_label)
        self.video_label.setFixedSize(640, 480)
        self.video_label.setAlignment(QtCore.Qt.AlignCenter)
        # 创建两个绘图控件，分别用于显示原始数据曲线和心率曲线
        self.raw_data_plot = PlotWidget()
        self.heart_rate_plot = PlotWidget()
        # 将绘图控件添加到布局中，并设置绘图控件的标题和坐标轴标签
        self.layout.addWidget(self.raw_data_plot)
        self.layout.addWidget(self.heart_rate_plot)
        self.raw_data_plot.setTitle("原始数据曲线")
        self.heart_rate_plot.setTitle("频率曲线")
        self.raw_data_plot.setLabel("left", "绿色像素均值")
        self.heart_rate_plot.setLabel("left", "频率幅度")
        #self.heart_rate_plot.setYRange(0,200)
        # 设置窗口的中央控件为一个小部件，并将布局设置为该小部件的布局
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.central_widget.setLayout(self.layout)

        self.data_len_max = 30*10

        # 在初始化方法中，创建一个变量，用于存储上一帧的时间
        self.last_frame_time = 0
        self.frame_rate = 30.0
        self.frame_rate_list = [30.0]

        self.freq_x = [0]
        self.freq_y = [0]

    # 定义一个开始视频的槽函数，用于启动定时器并设置定时器的信号和槽函数
    def start_video(self):
        #self.cap.open(0)
        # 启动定时器，每隔50毫秒触发一次timeout信号
        self.timer.start(20)
        # 将timeout信号连接到update_frame槽函数，用于更新视频帧和曲线数据
        self.timer.timeout.connect(self.update_frame)

    # 定义一个结束视频的槽函数，用于停止定时器并释放视频捕获对象
    def stop_video(self):
        # 停止定时器，并断开timeout信号和update_frame槽函数的连接
        self.timer.stop()
        self.timer.timeout.disconnect(self.update_frame)

    # 定义一个窗口关闭的事件，用于释放视频捕获对象
    def closeEvent(self, event):
        # 释放视频捕获对象
        self.cap.release()

    def smooth_signal(self,signal):
        # sigma is the standard deviation of the Gaussian kernel
        # it controls the degree of smoothing
        sigma = 3
        # apply the filter to the signal
        smoothed_signal = gaussian_filter1d(signal, sigma)
        # return the smoothed signal
        return smoothed_signal
    def calc_heart_rate(self,frame):
        # 将图像转换为灰度图，用于人脸检测
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 使用人脸检测器检测图像中的人脸，返回一个列表，每个元素是一个矩形区域
        faces = self.face_detector.detectMultiScale(gray, 1.3, 5)
        # 遍历每个检测到的人脸
        for (x, y, w, h) in faces:
            # 在原始图像上绘制一个绿色的矩形框，表示人脸区域
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # 截取人脸区域的图像，并转换为HSV颜色空间
            face = frame[y:y + h, x:x + w]
            hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
            # 计算人脸区域图像的绿色像素的均值，作为原始数据的一个点
            green_mean = np.mean(hsv[:, :, 1])
            # 将原始数据点添加到原始数据数组中，并保持数组的长度不超过100
            self.raw_data = np.append(self.raw_data, green_mean)
            self.filtered_data = self.raw_data
            if len(self.raw_data) > self.data_len_max:
                self.raw_data = self.raw_data[1:]

            if len(self.raw_data) > self.data_len_max//4:
                # 定义一个带通滤波器，参数为滤波器的阶数，截止频率，采样频率和类型
                b, a = butter(4, [0.8, 3], fs=self.frame_rate, btype="bandpass")
                # 使用filtfilt函数对原始数据进行滤波，返回一个滤波后的数组
                self.filtered_data = filtfilt(b, a, self.raw_data)

            # 使用傅里叶变换对原始数据进行频谱分析，得到频率和幅度
            freqs = np.fft.rfftfreq(len(self.filtered_data), d=1/self.frame_rate)
            amps = np.abs(np.fft.rfft(self.filtered_data))
            amps = self.smooth_signal(amps)
            self.freq_x = freqs*60
            self.freq_y = amps

            # 找到幅度最大的频率，作为心率的一个估计值，并转换为每分钟的次数
            heart_rate = np.max(freqs[np.argmax(amps)]) * 60
            # 将心率值添加到心率数据数组中，并保持数组的长度不超过100
            self.heart_rate_data = np.append(self.heart_rate_data, heart_rate)
            if len(self.heart_rate_data) > self.data_len_max//10:
                self.heart_rate_data = self.heart_rate_data[1:]


    # 定义一个更新视频帧和曲线数据的槽函数，用于读取摄像头的当前帧并进行处理和显示
    def update_frame(self):
        # 从视频捕获对象中读取一帧图像，返回一个布尔值和一个numpy数组
        ret, frame = self.cap.read()
        if not ret:
            return

            # 如果读取成功，继续处理图像
        self.calc_heart_rate(frame)

        # 在更新视频帧和曲线数据的槽函数中，计算当前帧的时间和帧率，并更新上一帧的时间
        current_frame_time = time.time()
        frame_rate = 1 / (current_frame_time - self.last_frame_time)
        if len(self.frame_rate_list)>10:
            self.frame_rate_list=self.frame_rate_list[1:]
        self.frame_rate_list.append(frame_rate)
        self.frame_rate= sum(self.frame_rate_list)/len(self.frame_rate_list)
        self.last_frame_time = current_frame_time

        # 在原始图像上绘制两个数字，分别表示帧率和心率，使用opencv的putText函数
        cv2.putText(frame, f"{self.frame_rate:.2f} fps", (480, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2)
        cv2.putText(frame, f"{self.heart_rate_data[-1]:.2f} bpm", (480, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2)

        # 将原始图像转换为Qt格式的图像，并设置标签的图片为该图像
        qt_image = QtGui.QImage(frame.data, frame.shape[1], frame.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
        self.video_label.setPixmap(QtGui.QPixmap.fromImage(qt_image))
        # 使用pyqtgraph绘制原始数据曲线和心率曲线，使用不同的颜色和样式
        self.raw_data_plot.clear()
        self.heart_rate_plot.clear()
        x=[ i/self.frame_rate for i in range(len(self.filtered_data))]
        self.raw_data_plot.plot(x, self.filtered_data,  pen="g")#symbol="-",

        self.heart_rate_plot.plot(self.freq_x, self.freq_y, pen="r")
        self.heart_rate_plot.plot([self.heart_rate_data[-1],self.heart_rate_data[-1]],[0,50], pen="r")

        # 创建一个应用对象，并传入命令行参数
if __name__ == "__main__":
    app = QApplication(sys.argv)
    # 创建一个窗口对象，并显示出来
    window = Ui_MainWindow()
    window.show()
    # 进入应用的主循环，并等待用户操作
    sys.exit(app.exec_())
