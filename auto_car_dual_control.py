import threading
import time
import cv2
import numpy as np

class LaneFollowing:
    def run(self):
        while True:
            # Simulate lane following logic
            print("Following lane...")
            time.sleep(1)  # Adjust sleep time for lane following frequency

class PersonDetection:
    def run(self):
        # Load YOLO model
        net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        
        cap = cv2.VideoCapture(0)  # Use the first camera
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            height, width, _ = frame.shape
            
            # Detecting objects
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)
            
            # Parsing detection results
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    
                    # Check if a person is detected
                    if confidence > 0.5 and class_id == 0: # Assuming class_id 0 is 'person'
                        print("Person detected!")
                        # Here you can add logic to take priority over lane following
                        time.sleep(0.5)  # Simulate some action on person detected
                        break  # Exit the loop upon detection
            
        cap.release()

lane_follower = LaneFollowing()
person_detector = PersonDetection()

# Create threads
lane_thread = threading.Thread(target=lane_follower.run)
person_thread = threading.Thread(target=person_detector.run)

# Start threads
lane_thread.start()
person_thread.start()

lane_thread.join()
person_thread.join()