import wx
import pytesseract
import os
import cv2
from PIL import Image
import numpy as np
import threading

APP_TITLE = u'识别群名并改名'
APP_ICON = 'res/python.ico'


class MainFrame(wx.Frame):

    def __init__(self, parent):
        wx.Frame.__init__(self, parent, -1, APP_TITLE)
        self.SetBackgroundColour(wx.Colour(255, 255, 255))
        self.SetSize((400, 160))
        self.Center()

        self.pathLabel = wx.StaticText(self, -1, u'文件路径：', pos=(10, 20), size=(60, -1), name='pathLabel', style=wx.TE_LEFT)
        self.path = wx.StaticText(self, -1, u'未选择', pos=(80, 20), size=(200, -1), name='path', style=wx.TE_LEFT)

        self.selectBtn = wx.Button(self, -1, u'选择', pos=(10, 50), size=(60, -1), style=wx.ALIGN_LEFT)
        self.startBtn = wx.Button(self, -1, u'开始', pos=(100, 50), size=(60, -1), style=wx.ALIGN_LEFT)

        self.tipLabel = wx.StaticText(self, -1, u'', pos=(10, 90), size=(200, -1), name='tipLabel', style=wx.TE_LEFT)
        self.tipCurtLabel = wx.StaticText(self, -1, u'', pos=(80, 90), size=(200, -1), name='tipCurtLabel', style=wx.TE_LEFT)

        self.Bind(wx.EVT_BUTTON, self.OnSelect, self.selectBtn)
        self.Bind(wx.EVT_BUTTON, self.OnStart, self.startBtn)

    def OnSelect(self, event):
        dlg = wx.DirDialog(self, u"选择文件夹", style=wx.DD_DEFAULT_STYLE)
        if dlg.ShowModal() == wx.ID_OK:
            print(dlg.GetPath())  # 文件夹路径
            self.path.SetLabelText(dlg.GetPath())
            self.tipLabel.SetLabelText("总共:"+str(len(os.listdir(self.path.GetLabelText()))))
        dlg.Destroy()

    def OnStart(self, event):
        if self.path.GetLabelText() == "未选择":
            msgDialog = wx.MessageDialog(parent=None, message = u"选择文件夹",style = wx.OK)
            if msgDialog.ShowModal() == wx.ID_OK:
                print(1)
            return

        self.selectBtn.Disable()
        self.startBtn.Disable()
        self.t1 = threading.Thread(target=self.OnStart2)
        self.t1.setDaemon(True)  # 设置为守护线程
        self.t1.start()

    def OnStart2(self):
        # if self.path.GetLabelText() == "未选择":
        #     msgDialog = wx.MessageDialog(parent=None, message = u"选择文件夹",style = wx.OK)
        #     if msgDialog.ShowModal() == wx.ID_OK:
        #        print(1)
        #     return
        # self.selectBtn.Disable()
        # self.startBtn.Disable()

        print(self.path.GetLabelText())
        fileNames = os.listdir(self.path.GetLabelText())

        if os.path.exists(self.path.GetLabelText()+"\\temp") == False:
            os.makedirs(self.path.GetLabelText()+"\\temp")

        for n in range(len(fileNames)):
            try:
                self.tipCurtLabel.SetLabelText("当前第"+str(n+1)+"个")
                fileName = fileNames[n]
                if fileName.endswith(('jpg','png','jpeg','bmp')) == False:
                    continue
                prefix = os.path.splitext(fileName)[0]
                suffix = os.path.splitext(fileName)[1]
                imgCount = self.ParseImg(self.path.GetLabelText(),fileName)

                fulltext = ""
                for i in range(imgCount):
                    image = self.cv_imread(self.path.GetLabelText()+"\\temp\\"+str(i)+".jpg")  # 打开图片
                    #image.load()  # 加载一下图片，防止报错，此处可省略
                    x, y = image.shape[0:2]
                    if x * y < 300 * 300:
                        image = cv2.resize(image, (y*5,x*5),cv2.INTER_LINEAR)
                    if x * y > 500 * 500:
                        image = cv2.resize(image, (y*2,x*2),cv2.INTER_LINEAR)
                    if  x * y > 300 * 300 and x * y < 500 * 500:
                        image = cv2.resize(image, (y*3,x*3),cv2.INTER_LINEAR)
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    _, dst = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
                    #均值模糊去噪方法。周围的都为均值 又称为低通滤波
                    #dst = cv2.blur(dst,(7,7))
                    #高斯模糊去噪方法。在某些情况下，需要对一个像素的周围的像素给予更多的重视
                    dst = cv2.GaussianBlur(dst, (3, 3), 1.1, None, 1.1)

                    #cv2.imshow('erosion', dst)
                    #cv2.waitKey()
                    #pytesseract.pytesseract.tesseract_cmd = 'G:/tesseract-4.0.0/tesseract-4.0.0-alpha/tesseract.exe'
                    #tessdata_dir_config = '--tessdata-dir "G:/tesseract-4.0.0/tesseract-4.0.0-alpha/tessdata" --psm 7 '
                    pytesseract.pytesseract.tesseract_cmd = 'tess/tesseract.exe'
                    tessdata_dir_config = '--tessdata-dir "tess/tessdata" --psm 7 '
                    text = pytesseract.image_to_string(dst, lang='chi_sim+eng', config=tessdata_dir_config)
                    print(text)
                    text = text.strip().replace("'","").replace(".","").replace("\"","").replace(" ","").replace(":","").replace(">","").replace("|","").replace("?","").replace("*","").replace("/"," ")
                    if i == 0:
                        fulltext = fulltext+text
                    if i == 1:
                        if text != "":
                            ddstr = "该二维码0天内(1月2日前)有效,重新进入将更新"
                            #numList = re.findall(r'\d+', text)
                            #for m in range(len(numList)):
                            #    text = text.replace(str(m),numList[m])
                            fulltext = fulltext+","+text
                if fulltext != "":
                    os.rename(self.path.GetLabelText()+"\\"+fileName, self.path.GetLabelText()+"\\"+fulltext+suffix)

                # 清除临时目录
                # filelist= os.listdir(self.path.GetLabelText()+"\\temp\\")
                # for f in filelist:
                #    filepath = os.path.join(self.path.GetLabelText()+"\\temp\\", f)
                #    if os.path.isfile(filepath):
                #      os.remove(filepath)
                #      print(str(filepath)+" removed!")
                if len(fileNames) == n+1:
                    self.tipCurtLabel.SetLabelText("处理完成")
                print("处理完成")
            except Exception as e:
                print(e)
                os.rename(self.path.GetLabelText()+"\\"+fileName, self.path.GetLabelText()+"\\"+prefix+"(不能识别)"+suffix)
        self.selectBtn.Enable()
        self.startBtn.Enable()

    def ParseImg(self, imagePath, imageName):
        image = Image.open(imagePath+"\\"+imageName)  # 打开图片
        image.load()
        img = self.cv_imread(imagePath+"\\"+imageName)

        regionp = self.ScanQrcodeRegion(imagePath,imageName,1)
        wp = regionp.get("w")
        hp = regionp.get("h")
        yp = regionp.get("y")
        xp = regionp.get("x")
        rate = 0.0
        if wp > hp:
            rate = hp/wp
        else:
            rate = wp/hp
        print("rate:"+str(rate))
        # img[int(yp-hp*0.3):int(yp+hp*0.3), xp:xp+wp
        ax = 0 if yp < int(hp*0.35) else yp - int(hp*0.35)
        by = image.height if yp+hp+int(hp*0.35) > image.height else yp+hp+int(hp*0.35)
        cropped_img = img[int(yp-hp*10):int(yp+hp+hp*10), xp:xp+wp] if rate < 0.6 else img[ax:by, xp:xp+wp]
        #cv2.imwrite(imagePath+'\\temp\\cropped_img.jpg', cropped_img ,[int(cv2.IMWRITE_JPEG_QUALITY),100])
        cv2.imencode('.jpg', cropped_img,[int(cv2.IMWRITE_JPEG_QUALITY),100])[1].tofile(imagePath+'\\temp\\cropped_img.jpg')

        # cropped_img1 = img[ax:hp, xp+int(wp/6):xp+wp]
        # cv2.imencode('.jpg', cropped_img1,[int(cv2.IMWRITE_JPEG_QUALITY),100])[1].tofile(imagePath+'\\temp\\0.jpg')
        # cropped_img1 = img[ax:hp, xp+int(wp/6):xp+wp]
        # cv2.imencode('.jpg', cropped_img1,[int(cv2.IMWRITE_JPEG_QUALITY),100])[1].tofile(imagePath+'\\temp\\1.jpg')

        img = self.cv_imread(imagePath+'\\temp\\cropped_img.jpg')
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #cv2.imshow('img',gray)
        #cv2.waitKey()
        # 此步骤形态学变换的预处理，得到可以查找矩形的图片
        # 参数：输入矩阵、输出矩阵数据类型、设置1、0时差分方向为水平方向的核卷积，设置0、1为垂直方向,ksize：核的尺寸
        sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize = 3)
        # cv2.imshow('sobel2',sobel)
        # cv2.waitKey()
        # 二值化
        ret, binary = cv2.threshold(sobel, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)
        # cv2.imshow('sobel',binary)
        # 设置膨胀和腐蚀操作的核函数
        element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 9))  #30, 9
        element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (24, 6))  #24, 6

        # 膨胀一次，让轮廓突出
        dilation = cv2.dilate(binary, element2, iterations = 1)
        # cv2.imshow('dilation', dilation)
        # cv2.waitKey()
        # 腐蚀一次，去掉细节，如表格线等。注意这里去掉的是竖直的线
        erosion = cv2.erode(dilation, element1, iterations = 1)
        #cv2.imshow('erosion', erosion)
        # cv2.waitKey()
        # aim = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,element1, 1 )   #此函数可实现闭运算和开运算
        # 以上膨胀+腐蚀称为闭运算，具有填充白色区域细小黑色空洞、连接近邻物体的作用

        # 再次膨胀，让轮廓明显一些
        dilation2 = cv2.dilate(erosion, element2, iterations = 3)

        #  查找轮廓
        contours, hierarchy = cv2.findContours(dilation2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # 利用以上函数可以得到多个轮廓区域，存在一个列表中。
        #  筛选那些面积小的
        regionList = []

        for i in range(len(contours)):
            # 遍历所有轮廓
            # cnt是一个点集
            cnt = contours[i]
            # 计算该轮廓的面积
            area = cv2.contourArea(cnt)
            # 面积小的都筛选掉、这个1000可以按照效果自行设置
            if area < 2000:
                continue
            # rect = cv2.minAreaRect(cnt)
            # box = np.int0(cv2.boxPoints(rect))
            # cv2.drawContours(img, [box], -1, (0, 255, 0), 3)
            # cv2.imshow('sobel2232',img)
            x, y, w, h = cv2.boundingRect(cnt)
            region1 = {"x":x,"y":y,"w":w,"h":h}
            regionList.append(region1)

        regionList.sort(key=lambda k: (k.get("y", 0)))



        regionp = self.ScanQrcodeRegion(imagePath,'temp\\cropped_img.jpg',2)
        uc = 0
        for j in range(len(regionList)):
            region = regionList[j]
            x = region.get("x")
            y = region.get("y")
            w = region.get("w")
            h = region.get("h")
            if self.isContain(region,regionp) == True:
                continue
            ttx = x if uc != 0 else x+int(regionp.get("w")/6)
            cropped = img[y:y+h, ttx:x+w]
            #cv2.imwrite(imagePath+'\\temp\\'+str(uc)+'.jpg', cropped ,[int(cv2.IMWRITE_JPEG_QUALITY),100])
            cv2.imencode('.jpg', cropped,[int(cv2.IMWRITE_JPEG_QUALITY),100])[1].tofile(imagePath+'\\temp\\'+str(uc)+'.jpg')
            uc = uc + 1
        return uc

    # 提取存在的二维码区域
    def ScanQrcodeRegion(self,imagePath, imageName,scanType):
        image = self.cv_imread(imagePath+"\\"+imageName)
        print(imagePath+"\\"+imageName)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        gradX = cv2.Sobel(gray, ddepth = cv2.CV_32F, dx = 1, dy = 0, ksize = -1)
        gradY = cv2.Sobel(gray, ddepth = cv2.CV_32F, dx = 0, dy = 1, ksize = -1)

        gradient = cv2.subtract(gradX, gradY)
        gradient = cv2.convertScaleAbs(gradient)

        #cv2.imshow("gradient",gradient)
        #原本没有过滤颜色通道的时候，这个高斯模糊有效，但是如果进行了颜色过滤，不用高斯模糊效果更好
        #blurred = cv2.blur(gradient, (9, 9))
        (_, thresh) = cv2.threshold(gradient, 225, 255, cv2.THRESH_BINARY)
        #cv2.imshow("thresh",thresh)
        #cv2.imwrite('thresh.jpg',thresh)
        qx, qy = image.shape[0:2]
        qrFactor = 15
        if qx * qy > 800 * 800 :
            qrFactor = 21
        if qx * qy < 500 * 500 :
            qrFactor = 11
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (qrFactor, qrFactor))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        #cv2.imshow("closed",closed)
        #cv2.imwrite('closed.jpg',closed)

        closed = cv2.erode(closed, None, iterations = 4)
        closed = cv2.dilate(closed, None, iterations = 4)
        #cv2.imwrite('closed1.jpg',closed)
        cnts, _ = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        c = sorted(cnts, key = cv2.contourArea, reverse = True)[0]
        if len(cnts) ==0:
            return {"x":0,"y":0,"w":0,"h":0}
        x, y, w, h = cv2.boundingRect(c)
        region1 = {"x":x,"y":y,"w":w,"h":h}
        # rect = cv2.minAreaRect(c)
        # box = np.int0(cv2.boxPoints(rect))
        # cv2.drawContours(image, [box], -1, (0, 255, 0), 3)
        # cv2.imshow("closedwuyz",image)
        # cv2.waitKey()
        return region1

    # 判断两个矩形区域是否相交
    def isContain(self,region1, region2):
        x1= region1.get("x")
        y1= region1.get("y")
        w1= region1.get("w")
        h1= region1.get("h")
        p1x = x1
        p1y = y1
        p2x = x1+w1
        p2y = y1+h1
        x2= region2.get("x")
        y2= region2.get("y")
        w2= region2.get("w")
        h2= region2.get("h")
        p3x = x2
        p3y = y2
        p4x = x2+w2
        p4y = y2+h2
        zx = abs(p1x + p2x - p3x - p4x)
        x = abs(p1x - p2x) + abs(p3x - p4x)
        zy = abs(p1y + p2y - p3y - p4y)
        y = abs(p1y - p2y) + abs(p3y - p4y)
        if zx <= x and zy <= y:
            return True
        else:
            return False

    ## 读取图像，解决imread不能读取中文路径的问题
    def cv_imread(self,filePath):
        cv_img=cv2.imdecode(np.fromfile(filePath,dtype=np.uint8),-1)
        ## imdecode读取的是rgb，如果后续需要opencv处理的话，需要转换成bgr，转换后图片颜色会变化
        ##cv_img=cv2.cvtColor(cv_img,cv2.COLOR_RGB2BGR)
        return cv_img

class MainApp(wx.App):
    def OnInit(self):
        self.SetAppName(APP_TITLE)
        self.Frame = MainFrame(None)
        self.Frame.Show()
        return True


if __name__ == "__main__":
    app = MainApp()
    app.MainLoop()
