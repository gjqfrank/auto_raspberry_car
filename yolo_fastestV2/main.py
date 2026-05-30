import enum

import cv2
import numpy as np
import requests
import time
from trafficlight import TrafficLight
from Car_state import Car_State

stream_url = "http://172.20.10.5:8080/?action=stream"
control_url = "http://172.20.10.5:5000/control"

cap = cv2.VideoCapture(stream_url)

last_command = "STOP"

car_state = Car_State.STOP

while True:
    ret, frame = cap.read()
    traffic_light = TrafficLight(frame, car_state)
    zebra_crossing = ZebraCrossing(frame, car_state)
    traffic_light.LIGHT_main()
    car_state = traffic_light.car_state
    if car_state == Car_State.STOP:
        requests.post(control_url, json={'command': "STOP"})
        continue
    elif car_state == Car_State.LANE_FOLLOW:
        zebra_crossing.ZEBRA_CROSSING_main()
        car_state = zebra_crossing.car_state
        if car_state == Car_State.WAIT:
            time.sleep(3)
        car_state = Car_State.LANE_FOLLOW
    elif car_state == Car_State.LANE_FOLLOW:
        requests.post(control_url, json={'command': "FORWARD"})
        

    cv2.imshow("frame", frame)



    if cv2.waitKey(1) == 27:
        break


cap.release()
cv2.destroyAllWindows()
