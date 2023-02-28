import os
import cv2
import numpy as np
import glob
from PIL import Image
import threading
import multiprocessing
#需要opencv和opencv-contrib-python


def del_file(path):#清空文件
    ls = os.listdir(path)
    for i in ls:
        c_path = os.path.join(path, i)
        if os.path.isdir(c_path):
            del_file(c_path)
        else:
            os.remove(c_path)
    print ('文件已经清空完成')



def getImageAndLabels(path):#给每张图片打标签
    facesSamples=[]
    ids=[]
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    print(imagePaths)
    face_detector= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    for imagePath in imagePaths:
        PIL_img=Image.open(imagePath).convert('L')
        img_numpy=np.array(PIL_img,'uint8')
        faces=face_detector.detectMultiScale(img_numpy,scaleFactor=1.5)

        id=int(os.path.split(imagePath)[1].split('.')[0])

        for x,y,w,h in faces:
            facesSamples.append(img_numpy[y:y+h,x:x+w])
            ids.append(id)
    return facesSamples,ids




def train():#训练
    name=input("请输入模型名称：")
    f1 = open("./data/trainname/names.txt", "a")
    f1.write("{}\n".format(name))
    f1.close()
    path = './data/image'
    faces, ids = getImageAndLabels(path)
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(ids))
    recognizer.write('trainer/{}.yml'.format(name))
    del_file('./data/image')
    del_file('./cap')




def videoface():#录入人脸信息时捕获摄像头
    print('开始')
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    i = 0
    photoname = 1

    while True:
        i = i + 1
        reg, frame = cap.read()

        cv2.imshow('window', frame)

        if i == 1:
            filename = str(photoname) + '.png'
            cv2.imwrite('cap' + '\\' + filename, frame)
            print(filename + '保存成功')
            i = 0

            photoname = photoname + 1
            if photoname >= 1000:  #训练张数
                break
        if cv2.waitKey(1) & 0xff == ord('q'):
            break

    cap.release()




flag=0
def videofacing():#识别过程中打开摄像头并捕获人脸
    print('开始')
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    i = 0
    photoname = 1
    global flag
    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    while True:

        ret, img = cap.read()

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=3)

        for (x, y, w, h) in faces:
            width = x + w
            height = y + h
            strok = 2
            color = (255, 0, 0)
            cv2.rectangle(img, (x, y), (width, height), color, strok)
        i = i + 1
        reg, frame = cap.read()

        cv2.imshow('face', img)

        if i == 50:

            filename = str(photoname) + '.png'
            cv2.imwrite('capturing' + '\\' + filename, frame)
            print("*摄像头正在捕获图片*")
            i = 0

            photoname = photoname + 1
            if photoname >= 2:
                 photoname = 1

        if cv2.waitKey(1) & 0xff == ord('q') or flag==1:
            flag=1
            break

    cap.release()
    cv2.destroyAllWindows()




def recognize():#根据训练集进行人脸识别
    names=[]
    trainnum = 0
    global flag
    f1 = open("./data/trainname/names.txt")
    for line in f1:
        line=line.strip('\n')
        names.append(line)
    f1.close()
    while 1:
        getface2()
        recognizer=cv2.face.LBPHFaceRecognizer_create()

        try:
            recognizer.read("./trainer/{}.yml".format(names[trainnum]))
            img=cv2.imread('./data/now/1.png')
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            face_detector= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            faces=face_detector.detectMultiScale(gray,scaleFactor=1.001)
            for x,y,w,h in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
                id,confidence=recognizer.predict(gray[y:y+h,x:x+w])
                print("*正在识别,训练集:{},标签ID:{},置信评分:{}".format(names[trainnum],id,confidence))
            cv2.imshow('result',img)
            del_file('./data/now/')
            if trainnum<len(names) and confidence>125:
               print("**暂时没有识别出来~")
               trainnum+=1

            elif confidence<=125:
               print("***识别为:{}***".format(names[trainnum]))
               flag=1
               break

        except:
            print("*暂时未检测到~")
            trainnum =0
            if flag==1:
              break




def getface():#捕获人脸之后进行面部裁剪以便训练数据
    size_m = 68
    size_n = 68

    def detect(img, cascade):
        rects = cascade.detectMultiScale(img, scaleFactor=1.5, minNeighbors=3, minSize=(100,100))
        if len(rects) == 0:
            return []
        rects[:, 2:] += rects[:, :2]
        return rects

    cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    imglist = glob.glob("cap\*.png")

    for list in imglist:

        img = cv2.imread(list)
        dst = img
        rects = detect(dst, cascade)
        for x1, y1, x2, y2 in rects:

            roi = dst[y1 :y2 , x1 :x2 ]
            img_roi = roi
            re_roi = cv2.resize(img_roi, (size_m, size_n))

            f = "{}/{}".format("./data", "image")
            print(f)
            if not os.path.exists(f):
                os.mkdir(f)

            path = list.split("\\")
            print(path)

            cv2.imwrite("{}/{}".format(f, path[1]), re_roi)




def getface2():#在识别的时候进行面部裁剪
        size_m = 68
        size_n = 68

        def detect(img, cascade):
            rects = cascade.detectMultiScale(img, scaleFactor=1.5, minNeighbors=5,minSize=(50,50))
            if len(rects) == 0:
                return []
            rects[:, 2:] += rects[:, :2]
            return rects

        cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        imglist = glob.glob("capturing/*.png")
        for list in imglist:

            img = cv2.imread(list)
            dst = img
            rects = detect(dst, cascade)
            for x1, y1, x2, y2 in rects:
                roi = dst[y1 :y2 , x1 :x2 ]
                img_roi = roi
                re_roi = cv2.resize(img_roi, (size_m, size_n))

                f = "{}/{}".format("./data", "now")
                print("捕获到图像{}".format(f))
                if not os.path.exists(f):
                    os.mkdir(f)

                path = list.split("\\")
                print("存到路径{}".format(path))

                cv2.imwrite("{}/{}".format(f, path[1]), re_roi)



def check():#多线程进行人脸检测
    del_file('./capturing/')
    p_video = threading.Thread(target=videofacing)
    p_rec = threading.Thread(target=recognize)
    p_video.start()
    p_rec.start()
    p_rec.join()
    p_video.join()



def main():

    x = input("请输入选项,输入Q录入新模型，输入W开始检测,输入E退出")
    if x.upper() == 'Q':
        videoface()
        getface()
        train()
    elif x.upper() == 'W':
        p = multiprocessing.Process(target=check)
        p.start()
        p.join()

    elif x.upper() == 'E':
        exit(0)




if __name__ == '__main__':
    while 1:
        main()




