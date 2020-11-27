import random

import wx
import os
import cv2
import numpy as np
import threading
import time
from pathlib import Path
from pyzbar import pyzbar
import tools.infer.utility as utility
import tools.infer.predict_system as ocr_sys

APP_TITLE = u'识别改名'
APP_ICON = 'res/python.ico'


# '道路识别'

class MainFrame(wx.Frame):

    def __init__(self, parent):
        wx.Frame.__init__(self, parent, -1, APP_TITLE)
        self.SetBackgroundColour(wx.Colour(255, 255, 255))
        self.SetSize((400, 160))
        self.Center()

        self.pathLabel = wx.StaticText(self, -1, u'文件路径：', pos=(10, 20), size=(60, -1), name='pathLabel',
                                       style=wx.TE_LEFT)
        self.path = wx.StaticText(self, -1, u'未选择', pos=(80, 20), size=(200, -1), name='path', style=wx.TE_LEFT)

        self.selectBtn = wx.Button(self, -1, u'选择', pos=(10, 50), size=(60, -1), style=wx.ALIGN_LEFT)
        self.startBtn = wx.Button(self, -1, u'开始', pos=(100, 50), size=(60, -1), style=wx.ALIGN_LEFT)

        self.tipLabel = wx.StaticText(self, -1, u'', pos=(10, 90), size=(200, -1), name='tipLabel', style=wx.TE_LEFT)
        self.tipCurtLabel = wx.StaticText(self, -1, u'', pos=(80, 90), size=(200, -1), name='tipCurtLabel',
                                          style=wx.TE_LEFT)

        self.Bind(wx.EVT_BUTTON, self.OnSelect, self.selectBtn)
        self.Bind(wx.EVT_BUTTON, self.OnStart, self.startBtn)

    def OnSelect(self, event):
        dlg = wx.DirDialog(self, u"选择文件夹", style=wx.DD_DEFAULT_STYLE)
        if dlg.ShowModal() == wx.ID_OK:
            print(dlg.GetPath())  # 文件夹路径
            self.path.SetLabelText(dlg.GetPath())
            self.tipLabel.SetLabelText("总共:" + str(len(os.listdir(self.path.GetLabelText()))))
        dlg.Destroy()

    def OnStart(self, event):
        if self.path.GetLabelText() == "未选择":
            msgDialog = wx.MessageDialog(parent=None, message=u"选择文件夹", style=wx.OK)
            if msgDialog.ShowModal() == wx.ID_OK:
                print(1)
            return

        self.selectBtn.Disable()
        self.startBtn.Disable()
        self.t1 = threading.Thread(target=self.OnStart2)
        self.t1.setDaemon(True)  # 设置为守护线程
        self.t1.start()

    def OnStart2(self):
        print(self.path.GetLabelText())

        args = utility.parse_args()
        args.image_dir = self.path.GetLabelText()
        args.det_model_dir = "./config/ch_det_mv3_db/"
        args.rec_model_dir = "./config/ch_rec_mv3_crnn/"
        args.use_gpu = False
        print("初始化系统")
        image_file_list = self.get_image_file_list(self.path.GetLabelText())
        text_sys = ocr_sys.TextSystem(args)
        print("完成初始化")
        imgFileDict = {}
        for image_file in image_file_list:
            imgFileDict[image_file] = image_file

        textNoDict = {}
        for n in range(len(image_file_list)):
            try:
                self.tipCurtLabel.SetLabelText("当前第" + str(n + 1) + "个")
                print(self.tipCurtLabel.GetLabelText())
                image_file = image_file_list[n]
                image_file = imgFileDict.get(image_file, "")
                if image_file.endswith(('jpg', 'png', 'jpeg', 'JPEG', 'JPG', 'bmp')) == False:
                    continue
                (filePath, filename) = os.path.split(image_file)
                (filePathAndName, ext) = os.path.splitext(image_file)

                img = self.cv_imread(image_file)
                if img is None:
                    print("error in loading image:{}".format(image_file))
                    continue
                img1 = self.cropImg(img)
                # cv2.imshow('img', img)
                # cv2.waitKey()
                startTime = time.time()
                txt_list = pyzbar.decode(img1)
                if len(txt_list) == 0:
                    img1 = cv2.resize(img1, (0, 0), fx=4, fy=4, interpolation=cv2.INTER_LINEAR)
                    txt_list = pyzbar.decode(img1)

                if len(txt_list) == 0:
                    continue

                finalText = str(txt_list[0].data, encoding="utf-8")
                # print(finalText)
                if finalText == "":
                    continue

                textArr = finalText.split(',')
                if len(textArr) < 7:
                    continue
                finalText = '-' + textArr[3] + '-' + textArr[6].split(".")[0]

                # 对受理方式进行ocr识别 先裁切图片
                img2 = self.cropImg2(img)
                # cv2.imshow('img2', img2)
                # cv2.waitKey()
                dt_boxes, rec_res = text_sys(img2)
                dt_num = len(dt_boxes)
                for dno in range(dt_num):
                    text, score = rec_res[dno]
                    if score >= 0.5:
                        text_str = "%s, %.3f" % (text, score)
                        # print(text_str)
                        if text.find("理号：") != -1:
                            index = text.find("理号：")
                            shouliNum = text[int(index + 3):]
                            print(shouliNum)
                            finalText = shouliNum + finalText

                elapse = time.time() - startTime
                print("Predict time of %s: %.3fs" % (image_file, elapse))

                # 对出现的字符串进行出现次数编号
                num = textNoDict.get(finalText, 0)
                if num == 0:
                    textNoDict[finalText] = 1
                else:
                    textNoDict[finalText] = (num + 1)
                print("finalText:", finalText, "append num:" + str(num))

                try:
                    if num == 0:
                        finalFileName = filePath + "\\" + finalText + ext
                        if finalFileName == image_file:
                            continue
                        file1 = Path(finalFileName)
                        if file1.exists():
                            newName = filePath + "\\" + finalText + self.getRandom(4) + ext
                            imgFileDict[finalFileName] = newName
                            os.rename(finalFileName, newName)
                        else:
                            os.rename(image_file, filePath + "\\" + finalText + ext)
                    else:
                        finalFileName = filePath + "\\" + finalText + "(" + str(num) + ")" + ext
                        if finalFileName == image_file:
                            continue
                        file1 = Path(finalFileName)
                        if file1.exists():
                            newName = filePath + "\\" + finalText + self.getRandom(4) + ext
                            imgFileDict[finalFileName] = newName
                            os.rename(finalFileName, newName)
                        else:
                            os.rename(image_file, filePath + "\\" + finalText + "(" + str(num) + ")" + ext)
                except Exception as e:
                    print(e)

                if len(image_file_list) == n + 1:
                    self.tipCurtLabel.SetLabelText("处理完成")
                print("处理完成")
            except Exception as e:
                print(e)

        self.selectBtn.Enable()
        self.startBtn.Enable()

    ## 读取图像，解决imread不能读取中文路径的问题
    def cv_imread(self, filePath):
        cv_img = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), -1)
        ## imdecode读取的是rgb，如果后续需要opencv处理的话，需要转换成bgr，转换后图片颜色会变化
        ##cv_img=cv2.cvtColor(cv_img,cv2.COLOR_RGB2BGR)
        # cv_img=Image.open(filePath).convert('RGBA')
        return cv_img

    def removeNum(self, a):
        b = []
        for i in a:
            if i not in "0123456789":
                b.append(i)
        return ("".join(b))

    def getRandom(self, n):
        s = ""
        for i in range(0, n):
            s = s + str(random.randint(0, 9))
        return s

    def get_image_file_list(self, img_file):
        imgs_lists = []
        if img_file is None or not os.path.exists(img_file):
            raise Exception("not found any img file in {}".format(img_file))

        img_end = ['jpg', 'png', 'jpeg', 'JPEG', 'JPG', 'bmp']
        if os.path.isfile(img_file) and img_file.split('.')[-1] in img_end:
            imgs_lists.append(img_file)
        elif os.path.isdir(img_file):
            for single_file in os.listdir(img_file):
                if single_file.split('.')[-1] in img_end:
                    imgs_lists.append(os.path.join(img_file, single_file))
        if len(imgs_lists) == 0:
            raise Exception("not found any img file in {}".format(img_file))
        return imgs_lists

    def cropImg(self, img):
        sp = img.shape
        height = sp[0]
        width = sp[1]
        ws = int(width * 0.8)
        hs = int(height * 0.25)
        cropped_img = img[0:hs, ws:width]

        # 裁剪出二维码区域
        gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)

        gradX = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        gradY = cv2.Sobel(gray, ddepth=cv2.CV_32F, dx=0, dy=1, ksize=-1)

        gradient = cv2.subtract(gradX, gradY)
        gradient = cv2.convertScaleAbs(gradient)

        # cv2.imshow("gradient",gradient)
        # 原本没有过滤颜色通道的时候，这个高斯模糊有效，但是如果进行了颜色过滤，不用高斯模糊效果更好
        # blurred = cv2.blur(gradient, (9, 9))
        (_, thresh) = cv2.threshold(gradient, 225, 255, cv2.THRESH_BINARY)
        # cv2.imshow("thresh",thresh)
        # cv2.imwrite('thresh.jpg',thresh)
        qx, qy = cropped_img.shape[0:2]
        qrFactor = 15
        if qx * qy > 800 * 800:
            qrFactor = 21
        if qx * qy < 500 * 500:
            qrFactor = 11
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (qrFactor, qrFactor))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        # cv2.imshow("closed",closed)
        # cv2.imwrite('closed.jpg',closed)

        closed = cv2.erode(closed, None, iterations=4)
        closed = cv2.dilate(closed, None, iterations=4)
        # cv2.imwrite('closed1.jpg',closed)
        cnts, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = sorted(cnts, key=cv2.contourArea, reverse=True)[0]
        if len(cnts) == 0:
            return cropped_img
        x, y, w, h = cv2.boundingRect(c)
        cropped = cropped_img[y:y + h, x:x + w]
        # draw_img = cv2.drawContours(cropped_img.copy(), c, -1, (0, 0, 255), 3)
        # cv2.imshow("draw_img", draw_img)
        # cv2.waitKey()
        return cropped

    def cropImg2(self, img):
        img2 = img.copy()
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        sp = img2.shape
        height = sp[0]
        width = sp[1]
        ws = int(width * 0.7)
        hs = int(height * 0.6)
        cropped_img2 = img2[hs:int(height * 0.9), int(0.3 * width):ws]
        return cropped_img2


class MainApp(wx.App):
    def OnInit(self):
        self.SetAppName(APP_TITLE)
        self.Frame = MainFrame(None)
        self.Frame.Show()
        return True


if __name__ == "__main__":
    app = MainApp()
    app.MainLoop()
