from cscore import CameraServer
from ntcore import NetworkTableInstance
import numpy as np
import cv2
from cones import ConePipeline
from cubes import CubePipeline
import math


class ContourData:
    con = None
    area = None

    def __init__(self, con, area) -> None:
        self.con = con
        self.area = area


def main():
    # ===================== CONSTANTS =====================

    input_width = 320
    input_height = 180

    input_half_width = int(input_width / 2)
    input_half_height = int(input_height / 2)

    camera_height_meters = 0.92964
    camera_pitch = math.radians(42)

    output_width = 320
    output_height = 180

    camera_matrix = np.array([[1418.385730011849546, 0.00, 920.8310160615919813],
                              [0.00, 1419.618382821758132,
                               514.8944628744075089],
                              [0.00000000000000000, 0.0000000000000000, 1.0000000000000000]])
    distortion_coefficients = np.array([[-4.423424326140373286e-01, 2.665611712922174026e-01,
                                         1.114619276960555905e-03, -2.671075682655178556e-04, -1.029716314732827681e-01]])

    fov_x, fov_y, _, _, _ = cv2.calibrationMatrixValues(cameraMatrix=camera_matrix, imageSize=[
        1920, 1080], apertureWidth=0.0029*1920, apertureHeight=0.0029*1080)

    max_camera_yaw = fov_x * math.pi/180/2
    max_camera_pitch = fov_y * math.pi/180/2
    focal_length_px_x = input_half_width / math.tan(max_camera_yaw)
    focal_length_px_y = input_half_width / math.tan(max_camera_pitch)

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
    # cube_pipeline = CubePipeline()

    while True:
        _, frame = input_stream.grabFrame(input_template)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame, -1)

        # frame = cv2.undistort(
        # frame, camera_matrix, distortion_coefficients, None, camera_matrix
        # )

        cone_pipeline.process(frame)
        cone_contours = cone_pipeline.find_contours_output

        # # cube_pipeline.process(frame)
        # # cube_contours = cube_pipeline.find_contours_output

        all_contours = cone_contours  # + cube_contours

        best_contour = None
        for i, con in enumerate(all_contours):
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

            yaw = math.atan2(cr_cone[0], focal_length_px_x)
            pitch = math.atan2(cr_cone[1], focal_length_px_y) - camera_pitch

            distance = camera_height_meters/math.tan(pitch)

            vision_nt.putNumber("yaw", yaw)
            vision_nt.putNumber("pitch", pitch)
            vision_nt.putNumber("distance", distance)
            vision_nt.putBoolean("detected", True)

        else:
            vision_nt.putBoolean("detected", False)
            vision_nt.putNumber("yaw", 0)
            vision_nt.putNumber("pitch", 0)
            vision_nt.putNumber("distance", 0)

        output_steam.putFrame(cv2.resize(frame, (output_width, output_height)))


if __name__ == '__main__':
    main()
