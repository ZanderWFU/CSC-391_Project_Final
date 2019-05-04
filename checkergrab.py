import cv2

DISPLAY_COLOR = False

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError("Cannot open webcam")

get_screen = False
screenshot_num = 0

while True:
    _, frame = cap.read()
    gframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gframe, (7, 6), None)
    if ret == True:
        corners2 = cv2.cornerSubPix(gframe, corners, (11, 11), (-1, -1), criteria)
        if not DISPLAY_COLOR:
            frame = cv2.cvtColor(gframe, cv2.COLOR_GRAY2BGR)
        frame = cv2.drawChessboardCorners(frame, (7, 6), corners2, ret)
        if get_screen:
            cv2.imwrite(f"Data/Calibration/pic{screenshot_num}.jpg", gframe)
            print(f"screenshot {screenshot_num} taken")
            screenshot_num += 1
            get_screen = False
    cv2.imshow("Input", frame)
    c = cv2.waitKey(1)
    if c == 27:
        break
    if c == ord('p'):
        print(f"Attempting screenshot {screenshot_num}")
        get_screen = True

cap.release()
cv2.destroyAllWindows()