import os
import time
import torch
from models.common import DetectMultiBackend
from utils.general import (
    check_img_size, non_max_suppression, scale_boxes)
from utils.torch_utils import select_device
from utils.augmentations import letterbox
from models.experimental import attempt_load

import numpy as np
import cv2
from sort import Sort
import imutils

class Detection:
    def __init__(self):
        self.weights = 'yolov5x.pt'
        self.imgsz = (1280, 1280)
        self.conf_thres = 0.5
        self.iou_thres = 0.45
        self.max_det = 1000
        self.device = 'cpu'
        self.classes = None
        self.agnostic_nms = True
        self.half = False
        self.dnn = False  # use OpenCV DNN for ONNX inference

    def _load_model(self):
        # Load model
        # self.device = select_device(self.device)
        if self.device == "cpu":
            arg = "cpu"
        else:
            arg = f"cuda:{self.device}"
        print(self.weights, self.classes, self.conf_thres, arg, self.imgsz)
        self.device = torch.device(arg)
        self.model = DetectMultiBackend(
            self.weights, device=self.device, dnn=self.dnn, fp16=self.half)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.imgsz = check_img_size(
            self.imgsz, s=self.stride)  # check image size

    def detect(self, image):
        image_copy = image.copy()
        bboxes = []
        im = letterbox(image, self.imgsz, stride=self.stride,
                       auto=self.pt)[0]  # resize
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)
        im = torch.from_numpy(im).to(self.device)
        im = im.half() if self.model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

        pred = self.model(im, augment=False, visualize=False)
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes,
                                   self.agnostic_nms, max_det=self.max_det)
        for i, det in enumerate(pred):
            if len(det):
                det[:, :4] = scale_boxes(
                    im.shape[2:], det[:, :4], image_copy.shape).round()
                for *xyxy, conf, cls in reversed(det):
                    x1, y1, x2, y2 = list(map(lambda x: max(0, int(x)), xyxy))
                    bboxes.append([int(x1), int(y1), int(x2), int(
                        y2), self.names[int(cls)], float(conf)])
                    # x_center = (x1 + x2 )//2
                    # y_center = (y1 + y2) //2
                    # cv2.circle(frame, (x_center, y_center), 2, (255, 255, 0), 2)

        return bboxes
point_marked= []
f = open('/Users/macbook/Dropbox/Mac/Documents/Yolov5/point_marked.txt', "r")
for line in f.readlines():
        slot = line.strip()
        x,y =  map(int,slot.split(","))
        point_marked.append([x,y])
# print(point_marked)
point_detect = []
f = open('/Users/macbook/Dropbox/Mac/Documents/Yolov5/detect.txt', "r")
for line in f.readlines():
        slot = line.strip()
        x,y = map(int,slot.split(","))
        point_detect.append([x,y])

if __name__ == '__main__':
    detector = Detection()
    detector.weights = r"yolov5x.pt"
    detector.classes = 2,5,7
    detector.conf_thres = 0.6
    detector.imgsz = (640, 640)
    detector.max_det = 1000
    detector.device = "cpu"
    detector.agnostic_nms = True
    detector.half = False
    detector.dnn = False
    detector._load_model()

path = '/Users/macbook/Dropbox/Mac/Documents/Yolov5/ch01_00000000000000000.mp4'
cap = cv2.VideoCapture(path)
m = 0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
print(fps)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, fps, (640,640))

while True:
    ret, frame = cap.read()
    img_detect = frame[point_detect[0][1]:point_detect[3][1], point_detect[0][0]:point_detect[3][0]]
    bboxes = list(detector.detect(img_detect))
    bboxes_new = sorted(bboxes, key = lambda x : x[0])
    # print(bboxes_new[0][0])
    # print(point_marked[0][0])

    point_bbox = []
    point_bbox.append([point_marked[0][0], bboxes_new[0][0]])
    for   bbox in bboxes_new:
        x,y,w,h, name, conf = bbox
        for i in range(0, len(bboxes_new)-1):
          if [bboxes_new[i][2], bboxes_new[i+1][0]] not in point_bbox:
            point_bbox.append([bboxes_new[i][2], bboxes_new[i+1][0]])
        cv2.rectangle(img_detect, (x, y), ( w, h), (0, 0, 255), 2)
    point_bbox.append([bboxes_new[0][0], point_marked[0][0]])
    
    point_bbox.append([bboxes_new[-1][2], point_marked[-1][0]])
    a = 0
    b = 0
    Lspace = []
    distance_marked_real = [7.5, 6.52, 6.76, 7.8, 5, 6.78, 5.32 ]
    for p in point_bbox:
        for i in range(0, len(point_marked)):
            if point_marked[i][0] <= p[0] < point_marked[i+1][0]:
                a = i
        for j in range(0, len(point_marked)):
            if point_marked[j][0] < p[1] <= point_marked[j+1][0]:
                b = j+1
        # print(point_marked[b][0]-point_marked[a][0])
        # print(point_marked[b][0]-point_marked[a][0])
        if point_marked[b][0]-point_marked[a][0] != 0:
            Lspace.append((p[1]-p[0])/(point_marked[b][0]-point_marked[a][0])*sum(distance_marked_real[a:b]))

    count_B = 0
    count_S = 0
    for i , bbox in enumerate(bboxes_new):
    # for i in range(0, len(Lspace)):
        if Lspace[i] > 0.1:
           cv2.rectangle(img_detect, (point_bbox[i][0], bboxes_new[i][1] - 20), (point_bbox[i][1], bboxes_new[i][3]+30), (135, 135, 255), 2)
           cv2.putText(img_detect, f'{round(Lspace[i],2)}m', (point_bbox[i][0],  bboxes_new[i][1]-50),cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 0), 3)
        if 4.2 <= Lspace[i] < 5.2:
            count_S += 1
            cv2.rectangle(frame,(point_bbox[i][0], 600),(point_bbox[i][0]+ 200,800), (225, 0, 0), 3)
            cv2.putText(frame, 'S', (point_bbox[i][0], 350),cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 0), 3)
        if  5.2 <= Lspace[i] <8.4:
            count_B += 1
            cv2.rectangle(img_detect,(point_bbox[i][0],  bboxes_new[i][1]),(point_bbox[i][0]+ (bboxes_new[i][2]-bboxes_new[i][0] + 50), bboxes_new[i][3]+20), (225, 0, 0), 3)
            cv2.putText(img_detect, 'B', (point_bbox[i][0], bboxes_new[i][1]),cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 0), 3)
        if 8.4 <= Lspace[i] < 10.4:
            count_S += 2
            cv2.rectangle(img_detect,(point_bbox[i][0], bboxes_new[i][1]),(point_bbox[i][0]+ bboxes_new[i][2]-bboxes_new[i][0]-50,bboxes_new[i][3]-50), (225, 0, 0), 3)
            cv2.putText(img_detect, 'S', (point_bbox[i][0], bboxes_new[i][1]),cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 0), 3)
            cv2.rectangle(img_detect,(point_bbox[i][0]+ (bboxes_new[i][2]-bboxes_new[i][0])+ 50,  bboxes_new[i][1] ),(point_bbox[i][0]+ (bboxes_new[i][2]-bboxes_new[i][0])+ (bboxes_new[i][2]-bboxes_new[i][0]) +50, bboxes_new[i][3]-50), (225, 0, 0), 3)
            cv2.putText(img_detect, 'S', (point_bbox[i][0]+300, bboxes_new[i][1]),cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 0), 3)
        if 10.4 <= Lspace[i] < 16.8:
            count_B += 2
            cv2.rectangle(img_detect,(point_bbox[i][0],  bboxes_new[i][1]),(point_bbox[i][0]+ (bboxes_new[i][2]-bboxes_new[i][0] + 50), bboxes_new[i][3]+20), (225, 0, 0), 3)
            cv2.putText(img_detect, 'B', (point_bbox[i][0], bboxes_new[i][1]+20),cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 0), 3)
            cv2.rectangle(img_detect,(point_bbox[i][0]+ (bboxes_new[i][2]-bboxes_new[i][0])+ 100,  bboxes_new[i][1] +50),(point_bbox[i][0]+ (bboxes_new[i][2]-bboxes_new[i][0])+ 150+(bboxes_new[i][2]-bboxes_new[i][0]), bboxes_new[i][3]+70), (225, 0, 0), 3)
            cv2.putText(img_detect, 'B', (point_bbox[i][0]+300, bboxes_new[i][1]+70),cv2.FONT_HERSHEY_SIMPLEX,1, (0, 255, 0), 3)
        # print(bbox[1],bbox[3])
    # img = cv2.resize(img,(1280,1280))

    # print(img.shape)
    cv2.rectangle(frame, (1700, 0), (1920, 100), (0,0,0), -1)
    cv2.putText(frame[0:100,frame.shape[1] -220:frame.shape[1]], f'Total: {count_B + count_S}', (30,20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255))
    cv2.putText(frame[0:100,frame.shape[1] -220:frame.shape[1]], f'B: {count_B }', (30,50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255))
    cv2.putText(frame[0:100,frame.shape[1] -220:frame.shape[1]], f'S: { count_S}', (30,80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255))



        # if 8.4 <= Lspace[i] < 10.4

    


    cv2.imshow('frame', frame)
    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
            break
out.release()
cap.release()
cv2.destroyAllWindows()