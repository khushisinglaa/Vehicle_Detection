import cv2              #importing OpenCV library
import numpy as np      #importing numpy

net = cv2.dnn.readNet('yolov3.weights','yolov3.cfg')
classes = []
with open("coco.names","r") as file:
    classes = [line.strip() for line in file.readlines()]


cap = cv2.VideoCapture('video.mp4')           # readng an input video with the help of opencv



# Get the video frame rate and dimensions
print(cap.get(cv2.CAP_PROP_FPS));
fps = cap.get(cv2.CAP_PROP_FOCUS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


car, truck, motorbike, bicycle, bus=0, 0, 0, 0,0

while True:
    _, img = cap.read()
    height, width, _ = img.shape
    # using blob function of openCV to preprocess
    blob = cv2.dnn.blobFromImage(img, 1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    # Detecting objects
    net.setInput(blob)#it takes 2 sec time for forward detection
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)
    # to show information on the screen
    boxes = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    # We use NMS function in OpenCV to perform Non-Maximum suppression.
    # we give it score threshold and nms threshold arguments.
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(boxes), 3))

    for i in indexes.flatten():
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        confidence = str(round(confidences[i], 2))
        color = colors[i]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)

        # cv2.rectangle(img, (x,y - 50), (x + w, y),(51,51,255), -1)
        cv2.putText(img, label + " " + confidence, (x, y-10), font, 1, (255,255,255),1)
        if label=="car":
            car=car+1;
            
        if label=="bus":
            bus=bus+1;
        if label=="motorbike":
            motorbike=motorbike+1;
        if label=="truck":
            truck=truck+1;
        if label=="bicycle":
            bicycle=bicycle+1;
        
    
    

    cv2.imshow('Output_Video', img)
 
    key = cv2.waitKey(1)
    if key == 27:
        break
print("car:", car);
print("truck:", truck);
print("bicycle:", bicycle);
print("bus:",bus);
print("motorbike:",motorbike);
cap.release()

cv2.destroyAllWindows()