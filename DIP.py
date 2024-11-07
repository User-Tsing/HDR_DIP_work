import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QFileDialog, QMessageBox, QDialog
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt, QSize, QStringListModel
import cv2
import numpy as np
import os
import math
import copy
import myFunction

#checked by STAssn
#python framework in pycharm


class showDCT(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("合成图像频谱展示")
        self.setGeometry(100, 100, 1100, 600)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.label1 = QLabel(self.central_widget)
        self.label2 = QLabel(self.central_widget)
        self.label1.setGeometry(10, 10, 500, 500)
        self.label2.setGeometry(520, 10, 500, 500)
        self.label1.setFixedSize(500, 500)
        self.label2.setFixedSize(500, 500)

    def show_dct(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转彩色RGB图
        height, width, depth = image.shape  # 图像长宽高
        bytes_per_line = width * 3
        q_image = QPixmap.fromImage(self.convert_to_qimage(image, width, height, bytes_per_line))
        self.label1.setPixmap(q_image.scaled(self.label1.size(), aspectRatioMode=True))

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        dct = cv2.dct(np.float32(gray))  # DCT变换
        dct = np.fft.fftshift(dct)  # 低频移到中间，然而真的要这样做吗？
        dct_magnitude = np.log(np.abs(dct) + 1)  # 频谱
        dct_magnitude = cv2.normalize(dct_magnitude, None, 0, 255, cv2.NORM_MINMAX)
        dct_magnitude = np.uint8(dct_magnitude)
        height, width = dct_magnitude.shape
        bytes_per_line = width
        q_image = QPixmap.fromImage(self.convert_to_qimage_gray(dct_magnitude, width, height, bytes_per_line))
        self.label2.setPixmap(q_image.scaled(self.label2.size(), aspectRatioMode=True))

    def convert_to_qimage_gray(self, image, width, height, bytes_per_line):
        return QImage(image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)

    def convert_to_qimage(self, image, width, height, bytes_per_line):
        return QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)

class HDRWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('HDR高动态范围图像重建')
        self.setGeometry(100, 80, 1700, 900)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)  # 创建中心窗口部件

        self.label1 = QLabel("原图1图像", self.central_widget)
        self.label2 = QLabel("原图2图像", self.central_widget)
        self.label3 = QLabel("原图3图像", self.central_widget)
        self.label4 = QLabel("原图4图像", self.central_widget)
        self.label5 = QLabel("原图1频谱", self.central_widget)
        self.label6 = QLabel("原图2频谱", self.central_widget)
        self.label7 = QLabel("原图3频谱", self.central_widget)
        self.label8 = QLabel("原图4频谱", self.central_widget)
        self.label9 = QLabel("合成图图像", self.central_widget)
        # self.label10 = QLabel("合成图频谱", self.central_widget)
        self.label11 = QLabel("HDR高动态图像合成重建", self.central_widget)

        font = QFont("Microsoft Yahei", 20)

        self.label1.setGeometry(20, 20, 200, 200)  # 标签控件设置，x,y,width,height
        self.label1.setFixedSize(200, 200)
        self.label5.setGeometry(230, 20, 200, 200)  # 标签控件设置，x,y,width,height
        self.label5.setFixedSize(200, 200)
        self.label2.setGeometry(20, 230, 200, 200)  # 标签控件设置，x,y,width,height
        self.label2.setFixedSize(200, 200)
        self.label6.setGeometry(230, 230, 200, 200)  # 标签控件设置，x,y,width,height
        self.label6.setFixedSize(200, 200)
        self.label3.setGeometry(20, 440, 200, 200)  # 标签控件设置，x,y,width,height
        self.label3.setFixedSize(200, 200)
        self.label7.setGeometry(230, 440, 200, 200)  # 标签控件设置，x,y,width,height
        self.label7.setFixedSize(200, 200)
        self.label4.setGeometry(20, 650, 200, 200)  # 标签控件设置，x,y,width,height
        self.label4.setFixedSize(200, 200)
        self.label8.setGeometry(230, 650, 200, 200)  # 标签控件设置，x,y,width,height
        self.label8.setFixedSize(200, 200)
        self.label9.setGeometry(900, 100, 750, 750)  # 标签控件设置，x,y,width,height
        self.label9.setFixedSize(750, 750)
        # self.label10.setGeometry(1400, 200, 450, 450)  # 标签控件设置，x,y,width,height
        # self.label10.setFixedSize(450, 450)
        self.label11.setGeometry(1000, 10, 600, 100)
        self.label11.setFont(font)
        self.folderpath = None
        self.savepath = None
        self.images_list = None
        self.window = None
        self.image_path = None

        self.button1 = QPushButton("导入图像", self.central_widget)
        self.button1.setGeometry(500, 200, 300, 100)
        self.button1.clicked.connect(self.loadFile)

        self.button2 = QPushButton("HDR图像合成", self.central_widget)
        self.button2.setGeometry(500, 400, 300, 100)
        self.button2.clicked.connect(self.hdr_process)

        self.button2 = QPushButton("合成图像频谱", self.central_widget)
        self.button2.setGeometry(500, 600, 300, 100)
        self.button2.clicked.connect(self.dct_show)

    def convert_to_qimage_gray(self, image, width, height, bytes_per_line):
        return QImage(image.data, width, height, bytes_per_line, QImage.Format_Grayscale8)

    def convert_to_qimage(self, image, width, height, bytes_per_line):
        return QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)


    def loadFile(self):  # 强行遍历法，一个一个显示图像
        folderpath = QFileDialog.getExistingDirectory(self, "选择图片所在文件夹")
        if folderpath:
            self.folderpath = folderpath
        else:
            self.folderpath = None
            QMessageBox.warning(self, "警告", "路径无效")
        self.images_list = [os.path.join(folderpath, it) for it in os.listdir(folderpath)
                            if it.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))][:4] #图像地址集群
        index = 0
        my_path = [""] * 4
        for image_path in self.images_list:
            if image_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                my_path[index] = image_path
                index += 1
                # print(index, self.images_list)
        if index == 4:
            image1 = cv2.imread(my_path[0])
            image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)  # 转彩色RGB图
            height, width, depth = image1.shape  # 图像长宽高
            bytes_per_line = width * 3
            q_image = QPixmap.fromImage(self.convert_to_qimage(image1, width, height, bytes_per_line))
            self.label1.setPixmap(q_image.scaled(self.label1.size(), aspectRatioMode=True))

            image2 = cv2.imread(my_path[1])
            image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)  # 转彩色RGB图
            height, width, depth = image2.shape  # 图像长宽高
            bytes_per_line = width * 3
            q_image = QPixmap.fromImage(self.convert_to_qimage(image2, width, height, bytes_per_line))
            self.label2.setPixmap(q_image.scaled(self.label2.size(), aspectRatioMode=True))

            image3 = cv2.imread(my_path[2])
            image3 = cv2.cvtColor(image3, cv2.COLOR_BGR2RGB)  # 转彩色RGB图
            height, width, depth = image3.shape  # 图像长宽高
            bytes_per_line = width * 3
            q_image = QPixmap.fromImage(self.convert_to_qimage(image3, width, height, bytes_per_line))
            self.label3.setPixmap(q_image.scaled(self.label3.size(), aspectRatioMode=True))

            image4 = cv2.imread(my_path[3])
            image4 = cv2.cvtColor(image4, cv2.COLOR_BGR2RGB)  # 转彩色RGB图
            height, width, depth = image4.shape  # 图像长宽高
            bytes_per_line = width * 3
            q_image = QPixmap.fromImage(self.convert_to_qimage(image4, width, height, bytes_per_line))
            self.label4.setPixmap(q_image.scaled(self.label4.size(), aspectRatioMode=True))

            gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
            dct1 = cv2.dct(np.float32(gray1))  # DCT变换
            dct1 = np.fft.fftshift(dct1)  # 低频移到中间，然而真的要这样做吗？
            dct1_magnitude = np.log(np.abs(dct1) + 1)  # 频谱
            dct1_magnitude = cv2.normalize(dct1_magnitude, None, 0, 255, cv2.NORM_MINMAX)
            dct1_magnitude = np.uint8(dct1_magnitude)
            height, width = dct1_magnitude.shape
            bytes_per_line = width
            q_image = QPixmap.fromImage(self.convert_to_qimage_gray(dct1_magnitude, width, height, bytes_per_line))
            self.label5.setPixmap(q_image.scaled(self.label5.size(), aspectRatioMode=True))

            gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
            dct2 = cv2.dct(np.float32(gray2))  # DCT变换
            dct2 = np.fft.fftshift(dct2)  # 低频移到中间，然而真的要这样做吗？
            dct2_magnitude = np.log(np.abs(dct2) + 1)  # 频谱
            dct2_magnitude = cv2.normalize(dct2_magnitude, None, 0, 255, cv2.NORM_MINMAX)
            dct2_magnitude = np.uint8(dct2_magnitude)
            height, width = dct2_magnitude.shape
            bytes_per_line = width
            q_image = QPixmap.fromImage(self.convert_to_qimage_gray(dct2_magnitude, width, height, bytes_per_line))
            self.label6.setPixmap(q_image.scaled(self.label6.size(), aspectRatioMode=True))

            gray3 = cv2.cvtColor(image3, cv2.COLOR_BGR2GRAY)
            dct3 = cv2.dct(np.float32(gray3))  # DCT变换
            dct3 = np.fft.fftshift(dct3)  # 低频移到中间，然而真的要这样做吗？
            dct3_magnitude = np.log(np.abs(dct3) + 1)  # 频谱
            dct3_magnitude = cv2.normalize(dct3_magnitude, None, 0, 255, cv2.NORM_MINMAX)
            dct3_magnitude = np.uint8(dct3_magnitude)
            height, width = dct3_magnitude.shape
            bytes_per_line = width
            q_image = QPixmap.fromImage(self.convert_to_qimage_gray(dct3_magnitude, width, height, bytes_per_line))
            self.label7.setPixmap(q_image.scaled(self.label7.size(), aspectRatioMode=True))

            gray4 = cv2.cvtColor(image4, cv2.COLOR_BGR2GRAY)
            dct4 = cv2.dct(np.float32(gray4))  # DCT变换
            dct4 = np.fft.fftshift(dct4)  # 低频移到中间，然而真的要这样做吗？
            dct4_magnitude = np.log(np.abs(dct4) + 1)  # 频谱
            dct4_magnitude = cv2.normalize(dct4_magnitude, None, 0, 255, cv2.NORM_MINMAX)
            dct4_magnitude = np.uint8(dct4_magnitude)
            height, width = dct4_magnitude.shape
            bytes_per_line = width
            q_image = QPixmap.fromImage(self.convert_to_qimage_gray(dct4_magnitude, width, height, bytes_per_line))
            self.label8.setPixmap(q_image.scaled(self.label8.size(), aspectRatioMode=True))

        elif index == 3:
            image1 = cv2.imread(my_path[0])
            image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)  # 转彩色RGB图
            height, width, depth = image1.shape  # 图像长宽高
            bytes_per_line = width * 3
            q_image = QPixmap.fromImage(self.convert_to_qimage(image1, width, height, bytes_per_line))
            self.label1.setPixmap(q_image.scaled(self.label1.size(), aspectRatioMode=True))

            image2 = cv2.imread(my_path[1])
            image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)  # 转彩色RGB图
            height, width, depth = image2.shape  # 图像长宽高
            bytes_per_line = width * 3
            q_image = QPixmap.fromImage(self.convert_to_qimage(image2, width, height, bytes_per_line))
            self.label2.setPixmap(q_image.scaled(self.label2.size(), aspectRatioMode=True))

            image3 = cv2.imread(my_path[2])
            image3 = cv2.cvtColor(image3, cv2.COLOR_BGR2RGB)  # 转彩色RGB图
            height, width, depth = image3.shape  # 图像长宽高
            bytes_per_line = width * 3
            q_image = QPixmap.fromImage(self.convert_to_qimage(image3, width, height, bytes_per_line))
            self.label3.setPixmap(q_image.scaled(self.label3.size(), aspectRatioMode=True))

            gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
            dct1 = cv2.dct(np.float32(gray1))  # DCT变换
            dct1 = np.fft.fftshift(dct1)  # 低频移到中间，然而真的要这样做吗？
            dct1_magnitude = np.log(np.abs(dct1) + 1)  # 频谱
            dct1_magnitude = cv2.normalize(dct1_magnitude, None, 0, 255, cv2.NORM_MINMAX)
            dct1_magnitude = np.uint8(dct1_magnitude)
            height, width = dct1_magnitude.shape
            bytes_per_line = width
            q_image = QPixmap.fromImage(self.convert_to_qimage_gray(dct1_magnitude, width, height, bytes_per_line))
            self.label5.setPixmap(q_image.scaled(self.label5.size(), aspectRatioMode=True))

            gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
            dct2 = cv2.dct(np.float32(gray2))  # DCT变换
            dct2 = np.fft.fftshift(dct2)  # 低频移到中间，然而真的要这样做吗？
            dct2_magnitude = np.log(np.abs(dct2) + 1)  # 频谱
            dct2_magnitude = cv2.normalize(dct2_magnitude, None, 0, 255, cv2.NORM_MINMAX)
            dct2_magnitude = np.uint8(dct2_magnitude)
            height, width = dct2_magnitude.shape
            bytes_per_line = width
            q_image = QPixmap.fromImage(self.convert_to_qimage_gray(dct2_magnitude, width, height, bytes_per_line))
            self.label6.setPixmap(q_image.scaled(self.label6.size(), aspectRatioMode=True))

            gray3 = cv2.cvtColor(image3, cv2.COLOR_BGR2GRAY)
            dct3 = cv2.dct(np.float32(gray3))  # DCT变换
            dct3 = np.fft.fftshift(dct3)  # 低频移到中间，然而真的要这样做吗？
            dct3_magnitude = np.log(np.abs(dct3) + 1)  # 频谱
            dct3_magnitude = cv2.normalize(dct3_magnitude, None, 0, 255, cv2.NORM_MINMAX)
            dct3_magnitude = np.uint8(dct3_magnitude)
            height, width = dct3_magnitude.shape
            bytes_per_line = width
            q_image = QPixmap.fromImage(self.convert_to_qimage_gray(dct3_magnitude, width, height, bytes_per_line))
            self.label7.setPixmap(q_image.scaled(self.label7.size(), aspectRatioMode=True))

        elif index == 2:
            image1 = cv2.imread(my_path[0])
            image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)  # 转彩色RGB图
            height, width, depth = image1.shape  # 图像长宽高
            bytes_per_line = width * 3
            q_image = QPixmap.fromImage(self.convert_to_qimage(image1, width, height, bytes_per_line))
            self.label1.setPixmap(q_image.scaled(self.label1.size(), aspectRatioMode=True))

            image2 = cv2.imread(my_path[1])
            image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)  # 转彩色RGB图
            height, width, depth = image2.shape  # 图像长宽高
            bytes_per_line = width * 3
            q_image = QPixmap.fromImage(self.convert_to_qimage(image2, width, height, bytes_per_line))
            self.label2.setPixmap(q_image.scaled(self.label2.size(), aspectRatioMode=True))

            gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
            dct1 = cv2.dct(np.float32(gray1))  # DCT变换
            dct1 = np.fft.fftshift(dct1)  # 低频移到中间，然而真的要这样做吗？
            dct1_magnitude = np.log(np.abs(dct1) + 1)  # 频谱
            dct1_magnitude = cv2.normalize(dct1_magnitude, None, 0, 255, cv2.NORM_MINMAX)
            dct1_magnitude = np.uint8(dct1_magnitude)
            height, width = dct1_magnitude.shape
            bytes_per_line = width
            q_image = QPixmap.fromImage(self.convert_to_qimage_gray(dct1_magnitude, width, height, bytes_per_line))
            self.label5.setPixmap(q_image.scaled(self.label5.size(), aspectRatioMode=True))

            gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
            dct2 = cv2.dct(np.float32(gray2))  # DCT变换
            dct2 = np.fft.fftshift(dct2)  # 低频移到中间，然而真的要这样做吗？
            dct2_magnitude = np.log(np.abs(dct2) + 1)  # 频谱
            dct2_magnitude = cv2.normalize(dct2_magnitude, None, 0, 255, cv2.NORM_MINMAX)
            dct2_magnitude = np.uint8(dct2_magnitude)
            height, width = dct2_magnitude.shape
            bytes_per_line = width
            q_image = QPixmap.fromImage(self.convert_to_qimage_gray(dct2_magnitude, width, height, bytes_per_line))
            self.label6.setPixmap(q_image.scaled(self.label6.size(), aspectRatioMode=True))

        else:
            QMessageBox.warning(self, "警告", "图像数量不足")

    def hdr_process(self):
        sequence = np.stack([cv2.imread(name) for name in self.images_list]) #拼接数据
        fused_results = myFunction.exposure_fusion(
            sequence,
            use_1=False,
            use_lappyr=True,
            use_gaussi=False,
            layers_num=7,
            alphas=(1.0, 1.0, 1.0),
            visual=False)  # 调用函数
        self.savepath = QFileDialog.getExistingDirectory(self, "选择图片所要保存的文件夹")
        for l, r in fused_results.items():  # 用于展示保存
            cv2.imwrite(os.path.join(self.savepath, l + ".png"), r, [cv2.IMWRITE_PNG_COMPRESSION, 0])

        new_image_path = os.path.join(self.savepath, "laplace_pyramid" + ".png") #新数据读取
        self.image_path = new_image_path
        processed_image = cv2.imread(new_image_path)  # warning:路径必须在项目根目录上否则无法显示
        processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)  # 转彩色RGB图
        height, width, depth = processed_image.shape  # 图像长宽高
        bytes_per_line = width * 3
        q_image = QPixmap.fromImage(self.convert_to_qimage(processed_image, width, height, bytes_per_line))
        self.label9.setPixmap(q_image.scaled(self.label9.size(), aspectRatioMode=True))

        # gray = cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY)
        # dct = cv2.dct(np.float32(gray))  # DCT变换
        # dct = np.fft.fftshift(dct)  # 低频移到中间，然而真的要这样做吗？
        # dct_magnitude = np.log(np.abs(dct) + 1)  # 频谱
        # dct_magnitude = cv2.normalize(dct_magnitude, None, 0, 255, cv2.NORM_MINMAX)
        # dct_magnitude = np.uint8(dct_magnitude)
        # height, width = dct_magnitude.shape
        # bytes_per_line = width
        # q_image = QPixmap.fromImage(self.convert_to_qimage_gray(dct_magnitude, width, height, bytes_per_line))
        # self.label10.setPixmap(q_image.scaled(self.label10.size(), aspectRatioMode=True))

        for l, r in fused_results.items():  # 用于展示保存
            myFunction.cv_show(r, l)

    def dct_show(self):
        self.window = None
        if self.window is None:
            self.window = showDCT()
            image = cv2.imread(self.image_path)
            self.window.show_dct(image)
        self.window.show()


class enhancedWindow(QMainWindow): #图像增强
    def __init__(self):
        super().__init__()

        self.setWindowTitle("图像增强")
        self.setGeometry(200, 200, 1000, 750)  # 窗口大小：x,y,width,height

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)  # 创建中心窗口部件

        self.label1 = QLabel("原图图像", self.central_widget)
        self.label2 = QLabel("原图频谱", self.central_widget)
        self.label3 = QLabel("增强图图像", self.central_widget)
        self.label4 = QLabel("增强图频谱", self.central_widget)

        self.label1.setGeometry(50, 50, 300, 300)  # 标签控件设置，x,y,width,height
        self.label1.setFixedSize(300, 300)
        self.label2.setGeometry(400, 50, 300, 300)  # 标签控件设置，x,y,width,height
        self.label2.setFixedSize(300, 300)
        self.label3.setGeometry(50, 400, 300, 300)  # 标签控件设置，x,y,width,height
        self.label3.setFixedSize(300, 300)
        self.label4.setGeometry(400, 400, 300, 300)  # 标签控件设置，x,y,width,height
        self.label4.setFixedSize(300, 300)

    def convert_to_qimage(self, image, width, height, bytes_per_line):
        return QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)

    def convert_to_qimage_gray(self, image, width, height, bytes_per_line):
        return QImage(image.data, width, height, bytes_per_line, QImage.Format_Grayscale8) #显示黑白图像（频谱等）

    def show_photo(self, image, image_dct, enhanced_image, enhanced_image_dct):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # 转彩色RGB图
        height, width, depth = image.shape  # 图像长宽高
        bytes_per_line = width * 3
        q_image = QPixmap.fromImage(self.convert_to_qimage(image, width, height, bytes_per_line))
        self.label1.setPixmap(q_image.scaled(self.label1.size(), aspectRatioMode=True))

        image_dct = cv2.normalize(image_dct, None, 0, 255, cv2.NORM_MINMAX)
        image_dct = np.uint8(image_dct)
        height, width = image_dct.shape
        bytes_per_line = width
        q_image = QPixmap.fromImage(self.convert_to_qimage_gray(image_dct, width, height, bytes_per_line))
        self.label2.setPixmap(q_image.scaled(self.label2.size(), aspectRatioMode=True))

        enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2RGB)  # 转彩色RGB图
        height, width, depth = enhanced_image.shape  # 图像长宽高
        bytes_per_line = width * 3
        q_image = QPixmap.fromImage(self.convert_to_qimage(enhanced_image, width, height, bytes_per_line))
        self.label3.setPixmap(q_image.scaled(self.label3.size(), aspectRatioMode=True))

        enhanced_image_dct = cv2.normalize(enhanced_image_dct, None, 0, 255, cv2.NORM_MINMAX)
        enhanced_image_dct = np.uint8(enhanced_image_dct)
        height, width = enhanced_image_dct.shape
        bytes_per_line = width
        q_image = QPixmap.fromImage(self.convert_to_qimage_gray(enhanced_image_dct, width, height, bytes_per_line))
        self.label4.setPixmap(q_image.scaled(self.label4.size(), aspectRatioMode=True))


class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("数字图像处理")
        self.setGeometry(100, 100, 1600, 900)  # 窗口大小：x,y,width,height

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)  # 创建中心窗口部件

        self.label1 = QLabel("点击按钮选择展示图像", self.central_widget)
        self.label1.setGeometry(100, 100, 600, 600)  # 标签控件设置，x,y,width,height
        self.label1.setFixedSize(600, 600)
        #self.label1.setStyleSheet("border: 1px solid black;")
        self.image = None  # 类变量，图像
        self.newWindow1 = None  # 新窗口，定义为无
        self.newWindow2 = None  # 新窗口，定义为无

        self.label_title = QLabel("数字图像处理", self.central_widget)
        font = QFont("Microsoft Yahei", 20)
        self.label_title.setFont(font)
        #self.label_title.setStyleSheet("background-color: rgb(128, 0, 255);")
        self.label_title.setGeometry(600, 10, 400, 80)

        self.label_tip = QLabel("checked by STAssn", self.central_widget)
        font2 = QFont("Microsoft Yahei", 7)
        self.label_tip.setFont(font2)
        self.label_tip.setStyleSheet("background-color: rgb(128, 0, 255);")
        self.label_tip.setGeometry(1450, 850, 140, 40)

        self.label2 = QLabel("此处展示图像频谱", self.central_widget)
        self.label2.setGeometry(900, 100, 600, 600)  # 标签控件设置，x,y,width,height
        self.label2.setFixedSize(600, 600)
        #self.label2.setStyleSheet("border: 1px solid black;")

        self.button1 = QPushButton("导入图像", self.central_widget)
        self.button1.setGeometry(100, 800, 200, 70)  # 按钮控件设置
        self.button1.clicked.connect(self.select_photo)  # 槽函数连接，只要函数名不要括号

        self.button2 = QPushButton("图像的傅里叶变换", self.central_widget)
        self.button2.setGeometry(350, 800, 200, 70)
        self.button2.clicked.connect(self.photo_fft)

        self.button3 = QPushButton("图像的离散余弦变换", self.central_widget)
        self.button3.setGeometry(350, 700, 200, 70)
        self.button3.clicked.connect(self.photo_dct)

        self.button4 = QPushButton("图像增强（高斯）", self.central_widget)
        self.button4.setGeometry(850, 800, 200, 70)
        self.button4.clicked.connect(self.photo_enhance)

        self.button4 = QPushButton("图像增强（中值）", self.central_widget)
        self.button4.setGeometry(850, 700, 200, 70)
        self.button4.clicked.connect(self.photo_enhance_Beta)

        self.button5 = QPushButton("图像加噪（高斯）", self.central_widget)
        self.button5.setGeometry(600, 800, 200, 70)
        self.button5.clicked.connect(self.photo_noisy)

        self.button6 = QPushButton("图像加噪（椒盐）", self.central_widget)
        self.button6.setGeometry(600, 700, 200, 70)
        self.button6.clicked.connect(self.photo_noisy_2)

        self.button7 = QPushButton("HDR高动态范围图像重建", self.central_widget)
        self.button7.setGeometry(1100, 800, 300, 70)
        self.button7.clicked.connect(self.HDR)


    def select_photo(self):
        options = QFileDialog.Options() #文件选择
        file_name, _ =QFileDialog.getOpenFileName(self, "选择图像文件", "", "Image files (*.jpg *.jpeg *.png *.bmp *.gif);;All Files (*)", options=options)
        if file_name:
            self.display_image1(file_name)
        else:
            QMessageBox.warning(self, "警告", "图片加载失败")

    def display_image1(self, file_name):
        #pixmap = QPixmap(file_name)
        #self.label1.setPixmap(pixmap.scaled(self.label1.size(), aspectRatioMode=True))
        self.image = cv2.imread(file_name) #warning:路径必须在项目根目录上否则无法显示
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB) #转彩色RGB图
        height, width, depth = self.image.shape #图像长宽高
        bytes_per_line = width * 3
        q_image = QPixmap.fromImage(self.convert_to_qimage(self.image, width, height, bytes_per_line))
        self.label1.setPixmap(q_image.scaled(self.label1.size(), aspectRatioMode=True))
        self.label2.setText(" ") #清除第二图

    def convert_to_qimage(self, image, width, height, bytes_per_line):
        return QImage(image.data, width, height, bytes_per_line, QImage.Format_RGB888)

    def photo_fft(self):
        #灰度图傅里叶变换并显示之
        if self.image is None:
            QMessageBox.warning(self, "警告", "请加载图像。")
            return
        if self.image is not None:
            #QMessageBox.warning(self, "警告", "傅里叶变换")
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            f = np.fft.fft2(gray) #二维傅里叶变换
            fshift = np.fft.fftshift(f) #频谱移动到中心
            magnitude_spectrum = 20 * np.log(np.abs(fshift)) #计算幅度谱（幅频特性）

            magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX) #归一化幅度谱
            magnitude_spectrum = np.uint8(magnitude_spectrum) #改为8位整型

            height, width = magnitude_spectrum.shape #提取频谱大小
            bytes_per_line = width #待定
            q_image = QPixmap.fromImage(self.convert_to_qimage_gray(magnitude_spectrum, width, height, bytes_per_line))
            self.label2.setPixmap(q_image.scaled(self.label2.size(), aspectRatioMode=True)) #显示频谱
            #QMessageBox.warning(self, "警告", "图片已显示")

    def convert_to_qimage_gray(self, image, width, height, bytes_per_line):
        return QImage(image.data, width, height, bytes_per_line, QImage.Format_Grayscale8) #显示黑白图像（频谱等）

    def photo_dct(self):
        if self.image is not None:
            gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            dct = cv2.dct(np.float32(gray)) #DCT变换
            dct = np.fft.fftshift(dct) #低频移到中间，然而真的要这样做吗？
            dct_magnitude = np.log(np.abs(dct)+1) #频谱
            dct_magnitude = cv2.normalize(dct_magnitude, None, 0, 255, cv2.NORM_MINMAX)
            dct_magnitude = np.uint8(dct_magnitude)
            height, width = dct_magnitude.shape
            bytes_per_line = width
            q_image = QPixmap.fromImage(self.convert_to_qimage_gray(dct_magnitude, width, height, bytes_per_line))
            self.label2.setPixmap(q_image.scaled(self.label2.size(), aspectRatioMode=True))

    def photo_enhance(self):
        if self.image is not None:
            image_o = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)  # 三色图
            for i in range(1, 2):  # 一次不够多来几次，然而真的是这样吗？两次吧再多不礼貌了
                image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB) #三色图
                b, g, r = cv2.split(image) #分通道处理
                denoised_image_b = cv2.GaussianBlur(b, (3, 3), 0) #高斯滤波器去噪
                denoised_image_g = cv2.GaussianBlur(g, (3, 3), 0)
                denoised_image_r = cv2.GaussianBlur(r, (3, 3), 0)
                denoised_image = cv2.merge((denoised_image_b, denoised_image_g, denoised_image_r))
                enhanced_image = self.equalize_image(denoised_image) #直方图均衡（库函数）
                original_dct = self.to_dct(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)) #DCT变换
                enhanced_dct = self.to_dct(cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)) #DCT变换
                output = 'to_charge.jpg'
                cv2.imwrite(output, enhanced_image)  # 存储图片
                self.image = cv2.imread(output)  # 更新操作对象
            self.display_image1(output)
            self.display_result(image_o, original_dct, enhanced_image, enhanced_dct)

    def equalize_image(self, image):
        chennels = cv2.split(image) #彩色图，划分信道（RGB）
        equalized_chennels = [] #创建数组
        for ch in chennels:
            #equalized_chennel = cv2.equalizeHist(ch) #按信道直方图均衡（库函数）
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # CLAHE均衡对象
            clahe_img = clahe.apply(ch)
            equalized_chennels.append(clahe_img) #均衡完加在后面
        return cv2.merge(equalized_chennels) #返回结果

    def to_dct(self, image):
        dct = cv2.dct(np.float32(image)) #DCT变换
        dct_magnitude = np.log(np.abs(dct)+1) #频谱
        dct_magnitude = np.fft.fftshift(dct_magnitude) #中心化
        return dct_magnitude

    def display_result(self, image, image_dct, enhanced_image, enhanced_image_dct):
        self.newWindow1 = None  # 清零，不然不重建
        if self.newWindow1 is None:
            self.newWindow1 = enhancedWindow()
            self.newWindow1.setWindowTitle("图像增强")
            self.newWindow1.show_photo(image, image_dct, enhanced_image, enhanced_image_dct)
        self.newWindow1.show()
        #newWindow.exec_()

    def enhanced_beta(self, image):
        b, g, r = cv2.split(image)
        b_process = self.channel_enhance(b)
        g_process = self.channel_enhance(g)
        r_process = self.channel_enhance(r)
        processed_image = cv2.merge((b_process, g_process, r_process))
        return processed_image

    def channel_enhance(self, channel):
        denoised = cv2.medianBlur(channel, 5) #中值滤波
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) #CLAHE均衡对象
        clahe_img = clahe.apply(denoised)
        return clahe_img

    def photo_enhance_Beta(self):
        if self.image is not None:
            image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            original = image
            processed_image = self.enhanced_beta(image)
            original_dct = self.to_dct(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))  # DCT变换
            enhanced_dct = self.to_dct(cv2.cvtColor(processed_image, cv2.COLOR_BGR2GRAY))  # DCT变换
            output = 'to_charge.jpg'
            cv2.imwrite(output, processed_image)  # 存储图片
            self.image = cv2.imread(output)  # 更新操作对象
            self.display_image1(output)
            self.display_result(original, original_dct, processed_image, enhanced_dct)

    def display_noise_result(self, image, image_dct, noisy_image, noisy_image_dct):
        self.newWindow1 = None  # 清零，不然不重建
        if self.newWindow1 is None:
            self.newWindow1 = enhancedWindow()
            self.newWindow1.setWindowTitle("图像加噪")
            self.newWindow1.show_photo(image, image_dct, noisy_image, noisy_image_dct)
        self.newWindow1.show()
        #newWindow.exec_()

    def photo_noisy(self): #高斯噪声生成器
        if self.image is not None:
            mean = 0
            sigma = 25
            image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            gaussian_noise = np.random.normal(mean, sigma, image.shape).astype(np.uint8)
            noisy_image = cv2.add(image, gaussian_noise)
            output = 'to_charge.jpg'
            cv2.imwrite(output, noisy_image) #存储图片
            original = image
            original_dct = self.to_dct(cv2.cvtColor(original, cv2.COLOR_BGR2GRAY))
            self.image = cv2.imread(output) #更新操作对象
            self.display_image1(output)
            noisy_dct = self.to_dct(cv2.cvtColor(noisy_image, cv2.COLOR_BGR2GRAY))
            self.display_noise_result(original, original_dct, noisy_image, noisy_dct)

    def photo_noisy_2(self): #椒盐噪声生成器
        if self.image is not None:
            salt_prob = 0.01
            pepper_prob = 0.01
            image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            noisy_image = np.copy(image)
            num_pepper = np.ceil(pepper_prob * image.size)
            coords = [np.random.randint(0, i-1, int(num_pepper)) for i in image.shape]
            noisy_image[coords[0], coords[1], :] = 0
            output = 'to_charge.jpg'
            cv2.imwrite(output, noisy_image)  # 存储图片
            original = image
            original_dct = self.to_dct(cv2.cvtColor(original, cv2.COLOR_BGR2GRAY))
            self.image = cv2.imread(output)  # 更新操作对象
            self.display_image1(output)
            noisy_dct = self.to_dct(cv2.cvtColor(noisy_image, cv2.COLOR_BGR2GRAY))
            self.display_noise_result(original, original_dct, noisy_image, noisy_dct)

    def HDR(self):
        self.newWindow2 = None
        if self.newWindow2 is None:
            self.newWindow2 = HDRWindow()
            #self.newWindow2.show()
        self.newWindow2.show()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())