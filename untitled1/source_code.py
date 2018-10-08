import datetime
import math
import cv2
import numpy as np

# global variables
width = 0
height = 0
EntranceCounter = 0
ExitCounter = 0
MinCountourArea = 3000  # Adjust ths value according to your usage
BinarizationThreshold = 70  # Adjust ths value according to your usage
OffsetRefLines = 150   # Adjust ths value according to your usage

drawing = False # true if mouse is pressed
mode = True # if True, draw rectangle. Press 'm' to toggle to curve
ix,iy = -1,-1





camera = cv2.VideoCapture(0)

# force 640x480 webcam resolution
camera.set(3, 640)
camera.set(4, 480)

ReferenceFrame = None

# The webcam maybe get some time / captured frames to adapt to ambience lighting. For this reason, some frames are grabbed and discarted.


while True:
    (grabbed, Frame) = camera.read()
    height = np.size(Frame, 0)
    width = np.size(Frame, 1)

    # if cannot grab a frame, this program ends here.
    if not grabbed:
        break

    # gray-scale convertion and Gaussian blur filter applying
    GrayFrame = cv2.cvtColor(Frame, cv2.COLOR_BGR2GRAY)
    GrayFrame = cv2.GaussianBlur(GrayFrame, (21, 21), 0)

    if ReferenceFrame is None:
        ReferenceFrame = GrayFrame
        continue

    # Background subtraction and image binarization
    FrameDelta = cv2.absdiff(ReferenceFrame, GrayFrame)
    FrameThresh = cv2.threshold(FrameDelta, BinarizationThreshold, 255, cv2.THRESH_BINARY)[1]

    # Dilate image and find all the contours
    FrameThresh = cv2.dilate(FrameThresh, None, iterations=2)
    _, cnts, _ = cv2.findContours(FrameThresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    QttyOfContours = 0

    # plot reference lines (entrance and exit lines)
##    rotate = cv2.getRotationMatrix2D(0,90,1)
    CoorXEntranceLine = (height // 2) - OffsetRefLines
    CoorXExitLine = (height // 2) + OffsetRefLines


    #cv2.line(Frame, (-640, 300), (width, CoorXEntranceLine), (255, 0, 0), 2)
    #cv2.line(Frame, (100, 250), (width, CoorXExitLine), (0, 0, 255), 2)

    def draw_circle(event, x, y, flags, param):
        global ix, iy, drawing, mode
        CoorXEntranceLine = y

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix, iy = x, y
            CoorXEntranceLine = y
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing == True:
                if mode == True:
                    curves[len(curves) - 1].append((x, y))  # append new points to the last list of curves
                else:
                    cv2.circle(Frame, (x, y), 5, (0, 0, 255), -1)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            if mode == True:
                curves.append([])  # adding a new list to curves
            else:
                cv2.circle(Frame, (x, y), 5, (0, 0, 255), -1)


    def draw_curves(myArray):

            for j in range(0, len(myArray)):

                 for i in range(1, len(myArray[j])):
                    cv2.line(Frame, myArray[j][i - 0], myArray[j][i], (0, 0, 255), 3)


    # check all found countours
    for c in cnts:
        # if a contour has small area, it'll be ignored
        if cv2.contourArea(c) < MinCountourArea:
            continue

        QttyOfContours = QttyOfContours + 1

        # draw an rectangle "around" the object
        (y, x, w, h) = cv2.boundingRect(c)
        cv2.rectangle(Frame, (y, x), (y + w, x + h), (0, 255, 0), 2)

        # find object's centroid
        CoordXCentroid = (y + x + w) // 2
        CoordYCentroid = (y + x + h) // 2
        ObjectCentroid = (CoordXCentroid, CoordYCentroid)
        cv2.circle(Frame, ObjectCentroid, 1, (0, 0, 0), 5)

        if (CheckEntranceLineCrossing(CoordYCentroid, CoorXEntranceLine)):
            if (CheckExitLineCrossing(CoorXExitLine)):
                EntranceCounter += 1

        if (CheckExitLineCrossing(CoordYCentroid, CoorXEntranceLine, CoorXExitLine)):
            ExitCounter += 1


    cv2.namedWindow('Original Frame')
    cv2.setMouseCallback('Original Frame', draw_circle)

    k = cv2.waitKey(1)
    curves = [[]]
    while (camera.isOpened()):

        (grabbed, Frame) = camera.read()

        draw_curves(curves)
        if Frame is None:
            break

        print
        "Total countours found: " + str(QttyOfContours)

        # Write entrance and exit counter values on frame and shows it
        cv2.putText(Frame, "Entrances: {}".format(str(EntranceCounter)), (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (250, 0, 1), 2)
        cv2.putText(Frame, "Exits: {}".format(str(ExitCounter)), (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)


        k = cv2.waitKey(1) & 0xFF
        cv2.imshow('Original Frame', Frame)

        if k == ord('m'):
            mode = not mode

        if k == ord("q"):
            break


    # cleanup the camera and close any open windows
    camera.release()
    cv2.destroyAllWindows()
