from ultralytics import YOLO
import cv2
import cvzone
import math

cap= cv2.VideoCapture("C:/Users/HP/Downloads/video.webm")
cap.set(3,1280)
cap.set(4,720)

model = YOLO("/FoodQualitiy.pt")


className = [
    "Apple",
    "Banana",
    "Bellpepper",
    "Bread",
    "Broccoli",
    "Cabbage",
    "Carrot",
    "Cauliflower",
    "Coriander",
    "Egg",
    "Grapes",
    "Kiwi",
    "MBanana",
    "Orange",
    "Pineapple",
    "Pomegranate",
    "Potato",
    "RApple",
    "RBanana",
    "RBread",
    "RCauliflower",
    "RCoriander",
    "RTomato",
    "Strawberry",
    "Tomato"
]



while True:
    success, img = cap.read()
    results = model(img, stream =True)
    for r in results:
        boxes = r.boxes
        for box in boxes:

            #Bounding Box
            x1, y1, x2, y2= box.xyxy[0]
            x1, y1, x2, y2= int(x1),int(y1), int(x2), int(y2)
             #print(x1, y1,x2,y2)
            cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)

            w,h = x2-x1, y2-y1
            cvzone.cornerRect(img,(x1,y1,w,h))

            # Confidence
            conf = math.ceil((box.conf[0]*100))/100
            cls = int(box.cls[0])
            cvzone.putTextRect(img, f'{className[cls]} {conf}', (max(0,x1), max(35, y1)),scale=1.5, thickness=1)
            # Class Name



    cv2.imshow("Webcam", img)
    cv2.waitKey(1)
