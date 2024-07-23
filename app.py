import streamlit as st
import cv2
from ultralytics import YOLO
import math
import os
from identifier import NoteIdentifier

model_file_path = 'model.pkl'
vit_model_name = 'google/vit-base-patch16-224-in21k'
target_class_name = "book"  # YOLO identifies banknotes as book
saved_image_name = 'note_image.jpg'
output_folder = "inference"
min_confidence = 0.20

# Create the "inference" folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Streamlit UI
st.title("Indian Banknote Identification")
st.text("Press 'Start' to begin the webcam feed and 'Stop' to end it.")

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

start_button = st.button('Start')
stop_button = st.button('Stop')

if start_button:
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    
    stframe = st.empty()
    
    while True:
        success, img = cap.read()
        if not success:
            st.error("Failed to capture image from webcam.")
            break
        
        results = model(img, stream=True)

        # coordinates
        for r in results:
            boxes = r.boxes

            for box in boxes:
                # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # convert to int values

                # confidence
                confidence = math.ceil((box.conf[0] * 100)) / 100

                # class name
                cls = int(box.cls[0])

                if cls < len(classNames) and classNames[cls] == target_class_name:
                    if confidence >= min_confidence:
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
                    else:
                        org = (50, 50)
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        fontScale = 1
                        color = (0, 0, 255)
                        thickness = 2
                        cv2.putText(img, "Confidence is too low", org, font, fontScale, color, thickness)

        stframe.image(img, channels="BGR")
        
        if stop_button:
            break
    
    cap.release()
    cv2.destroyAllWindows()
