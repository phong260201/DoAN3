import cv2

point_marked= []
f = open('/Users/macbook/Dropbox/Mac/Documents/Yolov5/point_marked.txt', "r")
for line in f.readlines():
        slot = line.strip()
        x,y =  map(int,slot.split(","))
        point_marked.append([x,y])
print(point_marked)

# img = cv2.imread('/Users/macbook/Dropbox/Mac/Documents/Yolov5/ch01_00000000006045200.jpg')

# for i in range(len(point_marked)) :
#         cv2.circle(img,point_marked[i], 10, (0,0,255), -1)
# distance_marked = []
# for i in range(len(point_marked) -1) :
#         cv2.line(img,point_marked[i], point_marked[i+1], (0,255, 0), 5)
#         distance_marked.append(point_marked[i+1][0]- point_marked[i][0])
# print(distance_marked)
# Lspace = []
# distance_marked_real = [7.5, 6.52, 6.76, 7.8, 5, 6.78, 5.32 ]

# cv2.waitKey(0)
point_detect= []
f = open('/Users/macbook/Dropbox/Mac/Documents/Yolov5/detect.txt', "r")
for line in f.readlines():
        slot = line.strip()
        x,y =  map(int,slot.split(","))
        point_detect.append([x,y])
print(point_detect)
