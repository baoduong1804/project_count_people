import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import*

model=YOLO('yolov8s.pt')

#Vẽ hai vùng khảo xác
# area1=[(312,388),(289,390),(474,469),(497,462)]
# area2=[(279,392),(250,397),(423,477),(454,469)]

area1=[(120,470),(90,510),(330,550),(350,505)]
area2=[(80,530),(50,570),(310,610),(330,570)]


def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

#Thêm video
cap=cv2.VideoCapture('video10.mp4')

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n") 
#print(class_list)

count=0
tracker = Tracker()

people_upping = {}# chứa các id và tọa độ người đi lên
people_downing = {}# chứa các id và tọa độ người đi xuống
upping = set()#chứa id người đi lên
downing = set()#chứa id người đi xuống

while True:    
    ret,frame = cap.read()
    if not ret:
        break
    count += 1
    if count % 2 != 0:
        continue
    #Kích thước video
    frame=cv2.resize(frame,(480 ,640))
    results=model.predict(frame)
    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")
    list=[]
             
    for index,row in px.iterrows():
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]
        if 'person' in c:
            list.append([x1,y1,x2,y2]) # thêm tọa độ người vào list
            
    bbox_id = tracker.update(list)
    for bbox in bbox_id :
        x3,y3,x4,y4,id = bbox
        
        ##people up
        results = cv2.pointPolygonTest(np.array(area2,np.int32),(((x4+x3)//2,y4)),False) #chấm tròn vào trong khu vực hai thì reutrn 1.0
        cv2.rectangle(frame,(x3,y3),(x4,y4),(0,255,0),2) # Vẽ hình chữ nhật
        cv2.circle(frame,((x4+x3)//2 ,y4),5,(255,0,255),-1) # vẽ chấm tròn nhỏ phía dưới khung
        cv2.putText(frame,str(id),(x3,y3),cv2.FONT_HERSHEY_COMPLEX,(0.5),(255,255,255),1)#Vẽ id người
        if results >= 0 :  # 1 > 0 => true
            people_upping[id] = (x4,y4)
            
        if id in people_upping :
            results1 = cv2.pointPolygonTest(np.array(area1,np.int32),(((x4+x3)//2,y4)),False)
            if results1 >= 0 :
                upping.add(id)
                
        ##people down
        results2 = cv2.pointPolygonTest(np.array(area1,np.int32),(((x4+x3)//2,y4)),False) #chấm tròn vào trong khu vực 1 thì reutrn 1.0
        if results2 >= 0 : 
            people_downing[id] = (x4,y4)
        if id in people_downing :
            results3 = cv2.pointPolygonTest(np.array(area2,np.int32),(((x4+x3)//2,y4)),False)
            if results3 >= 0 :
                downing.add(id)

            
            
        
    cv2.polylines(frame,[np.array(area1,np.int32)],True,(0,0,255),2)# Vẽ vùng khảo xác số 1
    cv2.putText(frame,str('1'),area1[0],cv2.FONT_HERSHEY_COMPLEX,(0.5),(0,255,0),1)# Vẽ số 1 lên góc vùng khảo xác số 1

    cv2.polylines(frame,[np.array(area2,np.int32)],True,(255,0,0),2)
    cv2.putText(frame,str('2'),area2[0],cv2.FONT_HERSHEY_COMPLEX,(0.5),(0,255,0),1)

    #print count people
    print(f"di len: {len(upping)}")
    print(f"di len: {len(downing)}")
    up = (len(upping))
    down =(len(downing))
    #Vẽ số người đi lên, xuống lên video
    cv2.putText(frame,str(f"Di len: {up}"),(20,40),cv2.FONT_HERSHEY_COMPLEX,(0.7),(0,255,0),2)
    cv2.putText(frame,str(f"Di xuong: {down}"),(20,80),cv2.FONT_HERSHEY_COMPLEX,(0.7),(0,255,0),2)
    
    
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1)&0xFF==27:
        break

cap.release()
cv2.destroyAllWindows()

