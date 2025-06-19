import cv2 #opencv
import urllib.request #to open and read URL
import numpy as np


url = 'http://192.168.1.10/cam-lo.jpg'
winName = 'ESP32 CAMERA'
cv2.namedWindow(winName,cv2.WINDOW_AUTOSIZE)

classNames = []
classFile = 'coco.names'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
#net.setInputSize(480,480)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

car_count = 0
person_count = 0
tracked_objects = {}
max_tracked_objects = 30
clear_threshold = 40
min_frames_static = 50

while(1):
    imgResponse = urllib.request.urlopen (url) # here open the URL
    imgNp = np.array(bytearray(imgResponse.read()),dtype=np.uint8)
    img = cv2.imdecode (imgNp,-1) #decodificamos

    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE) 


    classIds, confs, bbox = net.detect(img, confThreshold=0.5)

    if len(classIds) != 0:
        # Update tracked objects with detected ones
        new_tracked_objects = {}
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            class_name = classNames[classId - 1]
            object_id = (class_name, tuple(box))

            # Only track limited number of objects
            if len(tracked_objects) < max_tracked_objects:
                if object_id not in tracked_objects:
                    new_tracked_objects[object_id] = {
                        "clear_count": clear_threshold,
                        "static_frames": 0  # Initialize static frame count
                    }

        # Update existing and add new tracked objects
        tracked_objects.update(new_tracked_objects)

        # Check for clearing untracked objects
        for object_id, info in tracked_objects.items():
            if info["clear_count"] > 0:
                info["clear_count"] -= 1
                tracked_objects[object_id] = info
            else:
                del tracked_objects[object_id]

            if info["static_frames"] >= min_frames_static and object_id[0] in ("car", "person"):
                info["static_frames"] = 0  # Reset static frame count
                if object_id[0] == "car":
                    car_count += 1
                else:
                    person_count += 1

            else:
                info["static_frames"] += 1  # Increment static frame count
            tracked_objects[object_id] = info

        car_count = sum(1 for obj_id, _ in tracked_objects.items() if obj_id[0] == "car")
        person_count = sum(1 for obj_id, _ in tracked_objects.items() if obj_id[0] == "person")


    

    classIds, confs, bbox = net.detect(img,confThreshold=0.5)
    print(classIds,bbox)

    if len(classIds) != 0:
        for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
            cv2.rectangle(img,box,color=(0,255,0),thickness = 3) #mostramos en rectangulo lo que se encuentra
            cv2.putText(img, classNames[classId-1], (box[0]+10,box[1]+30), cv2.FONT_HERSHEY_COMPLEX, 1, (0,255,0),2)

    print("Number of cars detected:", car_count)
    print("Number of people detected:", person_count)

    cv2.imshow(winName,img) 
    tecla = cv2.waitKey(5) & 0xFF
    if tecla == 27:
        break

cv2.destroyAllWindows()