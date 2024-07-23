from ultralytics import YOLO
import cv2
import math 
import os
# import time

from identifier import NoteIdentifier

model_file_path = 'model.pkl'
vit_model_name='google/vit-base-patch16-224-in21k'
target_class_name = "book" # YOLO identifies banknotes as book
saved_image_name = 'note_image.jpg'
output_folder = "inference"

# Create the "inference" folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# model
model = YOLO("yolo-Weights\yolov8n-finetuned-riyal-RJ.pt")

# object classes
# classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
#               "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
#               "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
#               "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
#               "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
#               "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
#               "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
#               "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
#               "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
#               "teddy bear", "hair drier", "toothbrush"
#               ]

classNames=['100IndianRupees', '10IndianRupees', '200IndianRupees', '20IndianRupees', '500IndianRupees', '50IndianRupees', '5IndianRupees']


while True:
    success, img = cap.read()
    results = model(img, stream=True)

    # coordinates
    for r in results:
        boxes = r.boxes

        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) # convert to int values

            # confidence
            confidence = math.ceil((box.conf[0]*100))/100

            # class name
            cls = int(box.cls[0])

            if cls < len(classNames) and classNames[cls] == target_class_name:
                # draw a rectangle
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                # object details
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2

                # Save the image to the "inference" folder
                cropped_img = img[y1:y2, x1:x2]
                image_path = os.path.join(output_folder, saved_image_name)
                cv2.imwrite(image_path, cropped_img)

                # call the identifier model
                note_identifier = NoteIdentifier(pkl_file_path=model_file_path, 
                                                 vit_model_name=vit_model_name)
                predicted_class_str = note_identifier.get_prediction(image_path)

                # cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)
                cv2.putText(img, predicted_class_str, org, font, fontScale, color, thickness)

    cv2.imshow('Webcam', img)
    
    # wait for 2 m-seconds
    if cv2.waitKey(2) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()