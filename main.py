# from cscore import CameraServer
# from networktables import NetworkTables
import numpy
import cv2
from grip import GripPipeline


class ContourData:
    con = None
    area = None
    center = None
    perimeter = None

    def __init__(self, con, center, area, perimeter) -> None:
        self.con = con
        self.center = center
        self.area = area
        self.perimeter = perimeter


def main():
    left_threshold = 0.20
    right_threshold = 0.80

    screen_width = 480
    screen_height = 360

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
    cap.set(cv2.CAP_PROP_EXPOSURE, -7)

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
        ret, frame = cap.read()

        cv2.imshow('original', frame)

        pipeline.process(frame)
        contours = pipeline.find_contours_output

        contour_data = []

        for i, con in enumerate(contours):
            m = cv2.moments(con)
            cx = int(m["m10"] / m["m00"])
            cy = int(m["m01"] / m["m00"])
            center = (cx, cy)

            area = cv2.contourArea(con)
            perimeter = cv2.arcLength(con, True)

            contour_data.append(ContourData(con, center, area, perimeter))

        best_contour = None
        for data in contour_data:
            if best_contour == None:
                best_contour = data

            if data.area > best_contour.area:
                best_contour = data

        if not best_contour == None:
            con = best_contour.con
            center = best_contour.center
            area = best_contour.area
            perimeter = best_contour.perimeter

            cv2.drawContours(frame, [con], 0, (0, 255, 0), 3)
            cv2.circle(frame, center, 5, (0, 255, 0), 8)

        # cv2.putText(frame, str(area), center,
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        # cv2.putText(frame, str(perimeter), (center[0], center[1] + 50),
        #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        cv2.imshow('output', frame)

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
