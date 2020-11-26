import random

import wx
import os
import cv2
import numpy as np
import threading
import tools.infer.utility as utility
import tools.infer.predict_system as ocr_sys
import time
from pathlib import Path
import configparser

# from PIL import Image

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
        # fileNames = os.listdir(self.path.GetLabelText())

        # if os.path.exists(self.path.GetLabelText()+"\\temp") == False:
        #    os.makedirs(self.path.GetLabelText()+"\\temp")

        args = utility.parse_args()
        args.image_dir = self.path.GetLabelText()
        args.det_model_dir = "./config/ch_det_mv3_db/"
        args.rec_model_dir = "./config/ch_rec_mv3_crnn/"
        args.use_gpu = False
        print("初始化系统")
        config = configparser.ConfigParser()
        config.read("./config/config.ini", encoding='UTF-8')
        roadMapList = config.items('RoadWord')
        image_file_list = ocr_sys.get_image_file_list(args.image_dir)
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
                sp = img.shape
                height = sp[0]
                width = sp[1]
                # cv2.imshow('img', img)
                # cv2.waitKey()
                starttime = time.time()
                dt_boxes, rec_res = text_sys(img)
                elapse = time.time() - starttime
                print("Predict time of %s: %.3fs" % (image_file, elapse))
                dt_num = len(dt_boxes)

                # finalTextDict = {}
                finalText = ""
                for dno in range(dt_num):
                    text, score = rec_res[dno]
                    if score >= 0.5:
                        text_str = "%s, %.3f" % (text, score)
                        # print(text_str)
                        tempText = self.removeNum(text)
                        tempText2 = self.removeNum(finalText)
                        # finalTextDict[tempText] = finalText
                        if len(tempText) > len(tempText2):
                            finalText = text
                if finalText == "":
                    continue
                for i in range(len(roadMapList)):
                    roadWord = roadMapList[i][0]
                    if finalText.find(roadWord) != -1:
                        finalText = finalText.replace(roadWord, roadMapList[i][1])

                # finalText = finalTextDict.get(finalText, "")

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
                        finalFileName = filePath + "\\" + finalText + str(num) + ext
                        if finalFileName == image_file:
                            continue
                        file1 = Path(finalFileName)
                        if file1.exists():
                            newName = filePath + "\\" + finalText + self.getRandom(4) + ext
                            imgFileDict[finalFileName] = newName
                            os.rename(finalFileName, newName)
                        else:
                            os.rename(image_file, filePath + "\\" + finalText + str(num) + ext)
                except Exception as e:
                    print(e)
                    # os.rename(image_file, filePath+"/"+finalNo+"(标识码"+str(n)+")"+ext)

                # 清除临时目录
                # filelist = os.listdir(self.path.GetLabelText()+"\\temp\\")
                # for f in filelist:
                #    filepath = os.path.join(self.path.GetLabelText()+"\\temp\\", f)
                #    if os.path.isfile(filepath):
                #      os.remove(filepath)
                #      print(str(filepath)+" removed!")
                if len(image_file_list) == n + 1:
                    self.tipCurtLabel.SetLabelText("处理完成")
                print("处理完成")
            except Exception as e:
                print(e)
                # os.rename(image_file, filePathAndName+"(不能识别)"+ext)
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


class MainApp(wx.App):
    def OnInit(self):
        self.SetAppName(APP_TITLE)
        self.Frame = MainFrame(None)
        self.Frame.Show()
        return True


if __name__ == "__main__":
    app = MainApp()
    app.MainLoop()
