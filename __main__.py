from cscore import CameraServer
from ntcore import NetworkTableInstance
import numpy as np
import cv2
from cones import ConePipeline
import math


class ContourData:
    con = None
    area = None

    def __init__(self, con, area) -> None:
        self.con = con
        self.area = area


def main():
    # ===================== CONSTANTS =====================

    input_width = 180
    input_height = 90

    input_half_width = int(input_width / 2)
    input_half_height = int(input_height / 2)

    # camera_height_meters = 0.89535
    # camera_x_distance_meters = 0
    # camera_y_distance_meters = 0
    camera_pitch = math.radians(0)

    max_camera_yaw = math.radians(23.97)
    focal_length_px = input_half_width / math.tan(max_camera_yaw)

    output_width = 180
    output_height = 90

    # ==========================================
    
    camera = CameraServer.startAutomaticCapture("Camera", "/dev/video0")
    camera.setResolution(width=input_width, height=input_height)

    input_stream = CameraServer.getVideo()
    output_steam = CameraServer.putVideo(
        'Processed', width=input_width, height=input_height)

    input_template = np.zeros(
        shape=(input_height, input_width), dtype=np.uint8)

    ntinst = NetworkTableInstance.getDefault()

    ntinst.startClient4("visionprocessor")
    ntinst.setServerTeam(2264)
    ntinst.startDSClient()

    vision_nt = ntinst.getTable('ObjectVision')

    # readyStatus = vision_nt.getEntry("ready").getBoolean(False)
    # while not readyStatus:
    #     readyStatus = vision_nt.getEntry("ready").getBoolean(False)

    print("Running...")
    vision_nt.putBoolean("processing", True)

    cone_pipeline = ConePipeline()

    while True:
        _, frame = input_stream.grabFrame(input_template)

        cone_pipeline.process(frame)
        cone_contours = cone_pipeline.find_contours_output

        best_contour = None
        for i, con in enumerate(cone_contours):
            area = cv2.contourArea(con)

            if best_contour is None:
                best_contour = ContourData(con, area)

            if area > best_contour.area:
                best_contour = ContourData(con, area)

        if best_contour:
            con = best_contour.con

            M = cv2.moments(con)
            cone = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            cr_cone = (cone[0] - input_half_width, -(cone[1] -
                       input_half_height))  # center relative cone

            cv2.circle(frame, cone, 2, (255, 255, 255), 4)

            yaw = math.atan2(cr_cone[0], focal_length_px)
            pitch = math.atan2(cr_cone[1], focal_length_px) - camera_pitch

        # distance = camera_height_meters/math.tan(pitch)
        # robot_yaw = math.atan2(camera_y_distance_meters + distance * math.sin(camera_yaw), camera_x_distance_meters + distance * math.sin(camera_yaw))

        # cv2.putText(frame, f"YAW: {yaw * (180/math.pi)}", (10, 40),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        # cv2.putText(frame, f"PITCH: {pitch * (180/math.pi)}", (10, 90),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        # cv2.putText(frame, f"DISATNCE: {distance}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        # cv2.putText(frame, f"ROBOT_YAW: {robot_yaw}", (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            vision_nt.putNumber("yaw", yaw)
            vision_nt.putNumber("pitch", pitch)

        # vision_nt.putNumber("distance", distance)
        # vision_nt.putNumber("robot_yaw", robot_yaw)

        # cv2.imshow('frame', frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

        output_steam.putFrame(cv2.resize(frame, (output_width, output_height)))


if __name__ == '__main__':
    main()
