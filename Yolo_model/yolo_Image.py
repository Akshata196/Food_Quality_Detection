from ultralytics import YOLO
import cv2
import cvzone
import math

image = cv2.imread("C:/Users/HP/PycharmProjects/pythonProject/Image/509.jpeg")



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




results = model(image, stream =True)


for r in results:
  boxes = r.boxes
  for box in boxes:

            #Bounding Box
            x1, y1, x2, y2= box.xyxy[0]
            x1, y1, x2, y2= int(x1),int(y1), int(x2), int(y2)
             #print(x1, y1,x2,y2)
            cv2.rectangle(image,(x1,y1),(x2,y2),(255,0,255),3)

            w,h = x2-x1, y2-y1
            cvzone.cornerRect(image,(x1,y1,w,h))

            # Confidence
            conf = math.ceil((box.conf[0]*100))/100
            cls = int(box.cls[0])
            cvzone.putTextRect(image, f'{className[cls]} {conf}', (max(0,x1), max(35, y1)),scale=1.5, thickness=1)
            # Class Name


cv2.imshow("Image", image)
cv2.waitKey(30)