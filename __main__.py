# from cscore import CameraServer
# from ntcore import NetworkTableInstance
import numpy as np
import cv2
from grip import GripPipeline
import math


class ContourData:
    con = None
    area = None

    def __init__(self, con, area) -> None:
        self.con = con
        self.area = area


class LineData:
    p1 = None
    p2 = None
    length = None
    center = None

    def __init__(self, p1, p2) -> None:
        self.p1 = p1
        self.p2 = p2
        self.length = math.dist(p1, p2)
        self.center = (int((p1[0] + p2[0])/2), int((p1[1]+p2[1])/2))


def get_line_length(line):
    return line.length


# connectStatus = False


# def listener(connected):
#     print('; Connected=%s' % connected)
#     connectStatus = True


def main():
    # ===================== CAMERA CONSTANTS =====================

    focal_length_px = 1430
    camera_height_meters = 0.15875
    camera_x_distance_meters = 0.46 # (forwards) NEEDS MEASURING
    camera_y_distance_meters = 0.46 # (rightwards) NEEDS MEASURING

    input_width = 160
    input_height = 90
    
    input_cx = input_width / 2
    input_cy = input_height / 2

    output_width = 160
    output_height = 90

    # ==========================================

    # cserver = CameraServer

    # camera = cserver.startAutomaticCapture("Camera", "/dev/video0")

    # camera.setResolution(width=input_width, height=input_height)
    # camera.setBrightness(0)

    # input_stream = cserver.getVideo()

    # input_template = np.zeros(shape=(input_height, input_width),
    #                           dtype=np.uint8)

    # output = cserver.putVideo(
    #     'Processed', width=input_width, height=input_height)

    # ntinst = NetworkTableInstance.getDefault()

    # ntinst.startClient4("visionprocessor")
    # ntinst.setServerTeam(2264)
    # ntinst.startDSClient()

    # vision_nt = ntinst.getTable('ObjectVision')

    # readyStatus = vision_nt.getEntry("ready").getBoolean(False)

    # while not readyStatus:
    #     readyStatus = vision_nt.getEntry("ready").getBoolean(False)

    print("Running...")
    # vision_nt.putBoolean("processing", True)

    pipeline = GripPipeline()

    vid = cv2.VideoCapture(0)
    

    while True:
        # _, frame = input_stream.grabFrame(input_template)
        _, frame = vid.read()

        pipeline.process(frame)
        contours = pipeline.find_contours_output

        best_contour = None
        for i, con in enumerate(contours):
            area = cv2.contourArea(con)

            if best_contour is None:
                best_contour = ContourData(con, area)

            if area > best_contour.area:
                best_contour = ContourData(con, area)

        if best_contour:
            con = best_contour.con

            M = cv2.moments(con)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            cv2.circle(frame, (cx, cy), 10, (0, 0, 255), 20)

            print(cx, cy)

            camera_yaw = math.atan2(focal_length_px, cx - input_cx)
            pitch = math.atan2(focal_length_px, cy - input_cy)
            distance = camera_height_meters / math.tan(pitch)

            # robot_yaw = math.atan2(camera_y_distance_meters + distance * math.sin(camera_yaw), camera_x_distance_meters + distance * math.sin(camera_yaw))

            cv2.putText(frame, f"CAMERA_YAW: {camera_yaw * (180/math.pi)}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"PITCH: {pitch * (180/math.pi)}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, f"DISATNCE: {distance}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            # cv2.putText(frame, f"ROBOT_YAW: {robot_yaw}", (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            # vision_nt.putNumber("camera_yaw", camera_yaw)
            # vision_nt.putNumber("pitch", pitch)
            # vision_nt.putNumber("distance", distance)
            # vision_nt.putNumber("robot_yaw", robot_yaw)

        cv2.imshow('frame', frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # output.putFrame(cv2.resize(frame, (output_width, output_height)))

if __name__ == '__main__':
    main()
