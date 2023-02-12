# from cscore import CameraServer
# from networktables import NetworkTables
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


def main():
    # left_threshold = 0.20
    # right_threshold = 0.80

    screen_width = 640
    screen_height = 480

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
    cap.set(cv2.CAP_PROP_EXPOSURE, -4)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, screen_height)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, screen_width)

    # ----------------------------

    # cserver = CameraServer()
    # cserver.startAutomaticCapture()
    #
    # input_stream = cserver.getVideo()
    # output = cserver.putVideo('Processed', width=screen_width, height=screen_height)

    # zeros = numpy.zeros(
    #     shape=(screen_height, screen_width, 3), dtype=numpy.uint8)

    # NetworkTables.initialize(server='10.22.64.2')
    # vision_nt = NetworkTables.getTable('Vision')

    pipeline = GripPipeline()

    while True:
        _, frame = cap.read()

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

            hull = cv2.convexHull(con)
            hull_points = []
            for p in hull:
                hull_points.append((p[0][0], p[0][1]))

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

            cross_sections = [LineData(sides[0].p1, sides[1].p2), LineData(sides[0].p2, sides[1].p1)]
            cross_sections.sort(reverse=True, key=get_line_length)
            
            bottom_line = cross_sections[0]
            top_line = cross_sections[1]
            
            

            cv2.line(frame, bottom_line.p1, bottom_line.p2, (0, 0, 0), 4)
            cv2.line(frame, top_line.p1, top_line.p2, (0, 0, 0), 4)

            cv2.line(frame, bottom_line.p1, top_line.p1, (0, 0, 0), 4)
            cv2.line(frame, bottom_line.center, top_line.center, (0, 0, 0), 4)
            cv2.line(frame, bottom_line.p2, top_line.p2, (0, 0, 0), 4)

            cv2.circle(frame, top_line.p1, 5, (255, 255, 255), 8)
            cv2.circle(frame, top_line.center, 5, (255, 255, 255), 8)
            cv2.circle(frame, top_line.p2, 5, (255, 255, 255), 8)

            cv2.circle(frame, bottom_line.p1, 5, (255, 255, 255), 8)
            cv2.circle(frame, bottom_line.center, 5, (255, 255, 255), 8)
            cv2.circle(frame, bottom_line.p2, 5, (255, 255, 255), 8)

        cv2.imshow('final', frame)
        cv2.imshow('dilate', pipeline.cv_dilate_output)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # if len(output_data) > 0:
        #     detection_count = len(output_data)
        #
        #     for x in range(len(output_data)):
        #         (x, y), r = cv2.minEnclosingCircle(output_data[x])
        #         x = int(x)
        #         y = int(y)
        #         r = int(r)
        #
        #         if r > biggest_radius:
        #             biggest_radius = r
        #             best_detection = {"x": x, "y": y, "r": r}
        #
        # if best_detection:
        #     if 0 < best_detection["x"] < (screen_width * left_threshold):
        #         heading = -1
        #     elif best_detection["x"] > (screen_width * right_threshold):
        #         heading = 1
        #     else:
        #         heading = 2
        #
        #     # vision_nt.putNumber('heading', heading)
        #     # vision_nt.putNumber('x', best_detection["x"])
        #     # vision_nt.putNumber('y', best_detection["y"])
        #
        #     cv2.circle(img=output_image, center=(best_detection["x"], best_detection["y"]), radius=best_detection["r"], color=(0, 255, 0), thickness=5)
        #     print("x:", best_detection["x"], "y:", best_detection["y"], "r:", best_detection["r"], "heading:", heading)
        #
        # # vision_nt.putNumber("detectionCount", detection_count)
        # # output.putFrame(output_image)


if __name__ == '__main__':
    main()
