import sys
import cv2
import os
import numpy as np
from datetime import datetime
from analyzer import FrameAnalyzer
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QFileDialog, QMessageBox
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread

fa = FrameAnalyzer()

bar_val = 0
running = False
done = True
cuda = fa.detector.CUDA

input_path = "n.a"
video_path = "n.a"

def is_valid(path):
	cap = cv2.VideoCapture(path)
	open = cap.isOpened()
	cap.release()
	return open

class VideoThread(QThread):
	change_pixmap_signal = pyqtSignal(np.ndarray)

	def __init__(self):
		super().__init__()
		self._run_flag = True

	def run(self):
		global done, input_path

		done = False
		cap = cv2.VideoCapture(input_path)
		fa.reset()
		fa.pos_line = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) * 0.763)
		total = cap.get(cv2.CAP_PROP_FRAME_COUNT)

		print(fa.color_counts)

		if not cap.isOpened():
			print("cap is not open")
			return

		while self._run_flag:
			global bar_val, running

			if not running:
				continue
				cv2.waitKey(10)

			ret, cv_img = cap.read()
			if ret:
				fa.analyze(cv_img)
				self.change_pixmap_signal.emit(cv_img)

				pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
				bar_val = int(100*(pos/total))
			else:
				break
				
		# shut down capture system
		cap.release()

		path = 'output/'+str(datetime.now())+'.png'

		fa.plot_results(path=path)
		self.change_pixmap_signal.emit(cv2.imread(path))

		done = True
	
	def stop(self):
		"""Sets run flag to False and waits for thread to finish"""
		self._run_flag = False
		self.wait()
		

class MyThread(QThread):
	# Create a counter thread
	change_value = pyqtSignal(int)

	def run(self):
		while 1:
			self.change_value.emit(bar_val)
			cv2.waitKey(10)


class Ui_MainWindow(object):
	def setupUi(self, MainWindow):
		MainWindow.setObjectName("MainWindow")
		MainWindow.resize(1000, 800)
		self.centralwidget = QtWidgets.QWidget(MainWindow)
		self.centralwidget.setObjectName("centralwidget")
		self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.centralwidget)
		self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
		self.horizontalLayout_3.setSpacing(6)
		self.horizontalLayout_3.setObjectName("horizontalLayout_3")
		self.background_frame = QtWidgets.QFrame(self.centralwidget)
		self.background_frame.setStyleSheet("background-color:rgb(52,73,94)")
		self.background_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
		self.background_frame.setFrameShadow(QtWidgets.QFrame.Raised)
		self.background_frame.setObjectName("background_frame")
		self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.background_frame)
		self.verticalLayout_2.setObjectName("verticalLayout_2")
		self.horizontalLayout = QtWidgets.QHBoxLayout()
		self.horizontalLayout.setObjectName("horizontalLayout")
		self.label = QtWidgets.QLabel(self.background_frame)
		self.label.setObjectName("label")
		self.label.setMinimumWidth(800)
		self.horizontalLayout.addWidget(self.label)
		self.verticalLayout = QtWidgets.QVBoxLayout()
		self.verticalLayout.setObjectName("verticalLayout")
		self.progressBar = QtWidgets.QProgressBar(self.background_frame)
		self.progressBar.setStyleSheet("background-color:rgb(230,126,34);\n"
		"chunk{background-color:rgb(211,84,0)};\n"
		"")
		self.progressBar.setProperty("value", 24)
		self.progressBar.setObjectName("progressBar")
		self.verticalLayout.addWidget(self.progressBar)
		self.label_2 = QtWidgets.QLabel(self.background_frame)
		self.label_2.setStyleSheet("background-color:rgb(255, 255, 255);padding:10px")
		self.label_2.setObjectName("label_2")
		self.verticalLayout.addWidget(self.label_2)
		self.play_button = QtWidgets.QPushButton(self.background_frame)
		self.play_button.setStyleSheet("background-color:rgb(230,126,34);")
		self.play_button.setDefault(False)
		self.play_button.setObjectName("play_button")
		self.verticalLayout.addWidget(self.play_button)
		self.abort_button = QtWidgets.QPushButton(self.background_frame)
		self.abort_button.setStyleSheet("background-color:rgb(230,126,34);")
		self.abort_button.setObjectName("abort_button")
		self.verticalLayout.addWidget(self.abort_button)
		self.load_button = QtWidgets.QPushButton(self.background_frame)
		self.load_button.setStyleSheet("background-color:rgb(230,126,34);")
		self.load_button.setObjectName("load_button")
		self.verticalLayout.addWidget(self.load_button)
		self.checkBox = QtWidgets.QCheckBox(self.background_frame)
		self.checkBox.setStyleSheet("background-color:rgb(230,126,34);text-align:center")
		self.checkBox.setObjectName("checkBox")
		self.verticalLayout.addWidget(self.checkBox)
		self.horizontalLayout.addLayout(self.verticalLayout)
		self.verticalLayout_2.addLayout(self.horizontalLayout)
		self.horizontalLayout_3.addWidget(self.background_frame)
		MainWindow.setCentralWidget(self.centralwidget)
		self.menubar = QtWidgets.QMenuBar(MainWindow)
		self.menubar.setGeometry(QtCore.QRect(0, 0, 1273, 25))
		self.menubar.setObjectName("menubar")
		MainWindow.setMenuBar(self.menubar)
		self.statusbar = QtWidgets.QStatusBar(MainWindow)
		self.statusbar.setObjectName("statusbar")

		self.play_button.clicked.connect(self.togglePlay)
		self.abort_button.clicked.connect(self.abortAnalysis)
		self.load_button.clicked.connect(self.loadCapture)
		#MainWindow.setStatusBar(self.statusbar)

		self.retranslateUi(MainWindow)
		QtCore.QMetaObject.connectSlotsByName(MainWindow)

		self.thread = VideoThread()
		self.thread.change_pixmap_signal.connect(self.update_image)
		#self.thread.start()

		self.thread2 = MyThread()
		self.thread2.change_value.connect(self.setProgressVal)
		self.thread2.start()

		self.display_info()

	def togglePlay(self):
		global running, done, input_path

		# reset
		if done:
			# go for webcam
			if self.checkBox.isChecked():
				if is_valid(1):
					input_path = 1
				else:
					if is_valid(0):
						input_path = 0
					else:
						input_path = "n.a"
			else:
				input_path = video_path

			# video
			if input_path == "n.a":
				msgBox = QMessageBox()
				msgBox.setIcon(QMessageBox.Information)
				msgBox.setText("Video-Camera input is not specified or it is not valid. \nplease select video and try again")
				msgBox.setWindowTitle("Input Error")
				#msgBox.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
				returnValue = msgBox.exec()
				return

			self.thread = VideoThread()
			self.thread.change_pixmap_signal.connect(self.update_image)
			self.thread.start()

			running = True
			self.play_button.setText("Pause")
		# play/pause
		else:
			running = not running

			if running:
				self.play_button.setText("Pause")
			else:
				self.play_button.setText("Play")

	def abortAnalysis(self):
		global done, running
		done = True
		running = False
		self.play_button.setText("Restart")
		self.thread.stop()

	def loadCapture(self):
		filename, _ = QFileDialog.getOpenFileName(None, "Open Video", ".", "(*.mp4);;(*.avi)")
		global video_path
		video_path = filename

	def closeEvent(self, event):
		self.thread.stop()
		self.thread2.stop()
		event.accept()

	def retranslateUi(self, MainWindow):
		_translate = QtCore.QCoreApplication.translate
		MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
		self.label.setText(_translate("MainWindow", ""))
		self.label_2.setText(_translate("MainWindow", ""))
		self.play_button.setText(_translate("MainWindow", "Start"))
		self.abort_button.setText(_translate("MainWindow", "Abort and Analyze"))
		self.load_button.setText(_translate("MainWindow", "Select Video"))
		self.checkBox.setText(_translate("MainWindow", "Use Webcam"))

	def update_image(self, cv_img):
		"""Updates the image_label with a new opencv image"""
		qt_img = self.convert_cv_qt(cv_img)
		self.label.setPixmap(qt_img)
		self.display_info()

	def display_info(self):
		info = "Hello user we recommend you to use nvidia \ngpu with our app for improved peformance.\n"
		info +=	"If our app couldn't detect your GPU, We recom-\nmend you to update GPU drivers\n\n"
		info +=	"When video has finished or you aborted process \nanalysis result will be located under output\ndirectory \n\n"
		info += "Using GPU: " + str(cuda) + "\n" + "Running: " + str(running) + "\n\n"
		info += "Done: " + str(done) + "\n\n"
		info += "total cars: " + str(fa.detected) + "\n\n"

		colors = fa.color_counts
		for c in colors:
			info += str(c) + ": " + str(colors[c]) + "\n"

		self.label_2.setText(info)


	def convert_cv_qt(self, cv_img):
		"""Convert from an opencv image to QPixmap"""
		rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
		h, w, ch = rgb_image.shape
		bytes_per_line = ch * w
		convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)

		if w > 1000:
			rat = w / 1000
			w = 1000
			h = h / rat

		p = convert_to_Qt_format.scaled(int(w), int(h), Qt.KeepAspectRatio)
		return QPixmap.fromImage(p)

	def setProgressVal(self):
		self.progressBar.setValue(bar_val)

		if bar_val == 100:
			self.play_button.setText("Restart")

if __name__ == "__main__":
	print("loading app...")
	app = QtWidgets.QApplication(sys.argv)
	MainWindow = QtWidgets.QMainWindow()
	ui = Ui_MainWindow()
	ui.setupUi(MainWindow)
	MainWindow.show()
	sys.exit(app.exec_())
