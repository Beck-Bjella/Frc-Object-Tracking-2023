from cscore import CameraServer
# from networktables import NetworkTables

from ntcore import NetworkTableInstance
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


connectStatus = False


def listener(connected):
    print('; Connected=%s' % connected)
    connectStatus = True


def main():

    # # print(os.popen("lsof /dev/video0").read())
    input_screen_width = 160
    input_screen_height = 90

    # output_screen_width = 1280/4
    # output_screen_height = 720/4
    # output_screen_width = 320
    # output_screen_height = 180
    output_screen_width = 160
    output_screen_height = 90

    # cap = cv2.VideoCapture(0)

    # cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, screen_height)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, screen_width)

    # ----------------------------

    cserver = CameraServer

    camera = cserver.startAutomaticCapture("Camera", "/dev/video0")

    camera.setResolution(width=input_screen_width,
                         height=input_screen_height)

    input_stream = cserver.getVideo()

    input_template = np.zeros(shape=(input_screen_height, input_screen_width),
                              dtype=np.uint8)

    output = cserver.putVideo(
        'Processed', width=output_screen_width, height=output_screen_height)

    ntinst = NetworkTableInstance.getDefault()

    ntinst.startClient4("visionprocessor")
    ntinst.setServerTeam(2264)
    ntinst.startDSClient()

    # check if the connection was successful
    # ntinst.addConnectionListener(
    #     callback=listener, immediate_notify=True)

    # while (not connectStatus):
    #     pass

    print("about to do stuff")
    vision_nt = ntinst.getTable('ObjectVision')

    readyStatus = vision_nt.getEntry("ready").getBoolean(False)

    print("just did stuff, doing more soon")
    while (readyStatus):
        print("checking status")
        readyStatus = vision_nt.getEntry("ready").getBoolean(False)

    print("Got ready signal from robot, starting vision processing...")
    vision_nt.putBoolean("processing", True)

    pipeline = GripPipeline()

    while True:
        # _, frame = cap.read()
        frame_time, frame = input_stream.grabFrame(input_template)

        pipeline.process(frame)
        contours = pipeline.find_contours_output

        contour_data = []
        for i, con in enumerate(contours):
            area = cv2.contourArea(con)
            contour_data.append(ContourData(con, area))

        best_contour = None
        for data in contour_data:
            if best_contour == None:
                best_contour = data

            if data.area > best_contour.area:
                best_contour = data

        if not best_contour == None:
            con = best_contour.con
            area = best_contour.area

            M = cv2.moments(con)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            center_point = (cX, cY)

            hull = cv2.convexHull(con)
            hull_points = []
            for p in hull:
                hull_points.append((p[0][0], p[0][1]))

            cv2.drawContours(frame, [hull], -1, (255, 0, 0), 2)

            lines = []
            for i in range(len(hull_points)):
                first_point = hull_points[i]

                if i == len(hull_points) - 1:
                    second_point = hull_points[0]
                else:
                    second_point = hull_points[i + 1]

                lines.append(LineData(first_point, second_point))

            lines.sort(reverse=True, key=get_line_length)
            sides = lines[0:2]

            cross_sections = [LineData(sides[0].p1, sides[1].p2), LineData(
                sides[0].p2, sides[1].p1)]
            cross_sections.sort(reverse=True, key=get_line_length)

            bottom_line = cross_sections[0]
            top_line = cross_sections[1]

            cone = {"center": center_point, "top": top_line.center, "baseP1": bottom_line.p1,
                    "baseC": bottom_line.center, "baseP2": bottom_line.p2}
            angle = math.trunc(math.atan2(
                (cone["center"][1] - cone["top"][1]), -(cone["center"][0] - cone["top"][0])) * (180/math.pi))
            cone["angle"] = angle

            for key in cone:
                if (type(cone[key]) == tuple):
                    vision_nt.putNumberArray(key, cone[key])
                    # print(f"{key}: {cone[key]}")
                else:
                    vision_nt.putNumber(key, cone[key])
                    # print(f"{key}: {cone[key]}")

            cv2.arrowedLine(
                frame, cone["center"], cone["top"], (0, 0, 0), 4)

        output.putFrame(cv2.resize(
            frame, (output_screen_width, output_screen_height)))

        # cv2.imshow('final', frame)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break


if __name__ == '__main__':
    main()
