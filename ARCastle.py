import cv2
import numpy as np


#Getting texture
top_filename = "roof.jpg"
side_filename = "dark_wall.jpg"
front_filename = "door_wall.jpg"

top_texture = cv2.imread(f"Data/Textures/{top_filename}")
side_texture = cv2.imread(f"Data/Textures/{side_filename}")
front_texture = cv2.imread(f"Data/Textures/{front_filename}")

x, y = top_texture.shape[:2] #all textures will be the same shape
src_points = np.float32([[y-1,0], [0,0], [0,x-1], [y-1, x-1]])

#Prepare Screenshot Data
screen_f = open("Data/Screenshots/screen.txt", "r")
screenshot_num = int(screen_f.read())
screen_f.close()

neighbor_ref = [([[1,5], 3], [[0,4],1]), ([[0,4], 0], [[3,7],2]), ([[3,7], 1], [[2,6], 3]), ([[2,6], 2], [[1,5], 0])]

def drawCastle(img, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)
    side1_imgpts = np.array([imgpts[7], imgpts[6], imgpts[2], imgpts[3]])
    side1_dstpts = np.float32(side1_imgpts)
    side2_imgpts = np.array([imgpts[5], imgpts[4], imgpts[0], imgpts[1]])
    side2_dstpts = np.float32(side2_imgpts)
    front_imgpts = np.array([imgpts[4], imgpts[7], imgpts[3], imgpts[0]])
    front_dstpts = np.float32(front_imgpts)
    back_imgpts = np.array([imgpts[6], imgpts[5], imgpts[1], imgpts[2]])
    back_dstpts = np.float32(back_imgpts)
    top_imgpts = imgpts[4:]
    top_dstpts = np.float32(imgpts[4:])


    #determining the order of faces
    dside1 = (imgpts[6][0] - imgpts[7][0])**2 + (imgpts[6][1] - imgpts[7][1])**2
    dside2 = (imgpts[4][0] - imgpts[5][0])**2 + (imgpts[4][1] - imgpts[5][1])**2
    dfront = (imgpts[4][0] - imgpts[7][0])**2 + (imgpts[4][1] - imgpts[7][1])**2
    dback = (imgpts[6][0] - imgpts[5][0])**2 + (imgpts[6][1] - imgpts[5][1])**2

    face_dict = {dside2:0, dfront:1, dside1:2, dback:3}
    closest = face_dict[max(dside2, dfront, dside1, dback)]
    right_edge = neighbor_ref[closest][0][0]
    left_edge = neighbor_ref[closest][0][0]
    dright = (imgpts[right_edge[0]][0] - imgpts[right_edge[1]][0])**2 + (imgpts[right_edge[0]][1] - imgpts[right_edge[1]][1])**2
    dleft = (imgpts[left_edge[0]][0] - imgpts[left_edge[1]][0])**2 + (imgpts[left_edge[0]][1] - imgpts[left_edge[1]][1])**2

    if right_edge >= left_edge:
        face_list = [neighbor_ref[closest][1][1], neighbor_ref[closest][0][1], closest]
    else:
        face_list = [neighbor_ref[closest][0][1], neighbor_ref[closest][1][1], closest]
    for wall in face_list:
        #side2
        if wall is 0:
            img = cv2.drawContours(img, [side2_imgpts], -1, (0, 0, 0), -3)
            projective_matrix = cv2.getPerspectiveTransform(src_points, side2_dstpts)
            img_output = cv2.warpPerspective(side_texture, projective_matrix, (img.shape[1], img.shape[0]))
            img = cv2.addWeighted(img, 1, img_output, 1, 0)
        #front
        elif wall is 1:
            img = cv2.drawContours(img, [front_imgpts], -1, (0, 0, 0), -3)
            projective_matrix = cv2.getPerspectiveTransform(src_points, front_dstpts)
            img_output = cv2.warpPerspective(front_texture, projective_matrix, (img.shape[1], img.shape[0]))
            img = cv2.addWeighted(img, 1, img_output, 1, 0)
        #side1
        elif wall is 2:
            img = cv2.drawContours(img, [side1_imgpts], -1, (0, 0, 0), -3)
            projective_matrix = cv2.getPerspectiveTransform(src_points, side1_dstpts)
            img_output = cv2.warpPerspective(side_texture, projective_matrix, (img.shape[1], img.shape[0]))
            img = cv2.addWeighted(img, 1, img_output, 1, 0)
        #back
        elif wall is 3:
            img = cv2.drawContours(img, [back_imgpts], -1, (0, 0, 0), -3)
            projective_matrix = cv2.getPerspectiveTransform(src_points, back_dstpts)
            img_output = cv2.warpPerspective(front_texture, projective_matrix, (img.shape[1], img.shape[0]))
            img = cv2.addWeighted(img, 1, img_output, 1, 0)

    #drawing roof
    img = cv2.drawContours(img, [top_imgpts],-1,(0,0,0),-3)
    projective_matrix = cv2.getPerspectiveTransform(src_points, top_dstpts)
    img_output = cv2.warpPerspective(top_texture, projective_matrix, (img.shape[1], img.shape[0]))
    img = cv2.addWeighted(img, 1, img_output, 1, 0)



    return img

#using camera callibration stuff
with np.load('matrixParams_mbp.npz') as X:
    mtx, dist, _, _ = [X[i] for i in ('mtx', 'dist', 'rvecs', 'tvecs')]
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((6*7, 3), np.float32)
objp[:,:2] = np.mgrid[0:7, 0:6].T.reshape(-1, 2)

#parameters for the castle
i = 3 #bottom face side lengths
i2 = 3 #top face side length
tall = 3 #height of the castle
axis = np.float32([[0,0,0], [0,i,0], [i,i,0], [i,0,0],
                   [0,0,-tall], [0,i2,-tall], [i2,i2,-tall], [i2,0,-tall]])

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise IOError("Cannot open webcam")
get_screen = False
while True:
    _, frame = cap.read()
    gframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gframe, (7, 6), None)
    if ret == True:
        #refine the corners to more accurate coordinants
        corners2 = cv2.cornerSubPix(gframe, corners, (11,11), (-1,-1), criteria)

        # Find the rotation and translation vectors.
        _, rvecs, tvecs, _ = cv2.solvePnPRansac(objp, corners2, mtx, dist)

        # project the model 3D points to the image plane
        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)

        frame = drawCastle(frame, imgpts)
        # attempting to take a screenshot
        if get_screen:
            cv2.imwrite(f"Data/Screenshots/pic{screenshot_num}.jpg", frame)
            print(f"screenshot {screenshot_num} taken")
            cv2.imshow("Screenshot", frame)
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

#updating screenshot info
screen_f = open("Data/Screenshots/screen.txt", "w")
screen_f.write(str(screenshot_num))
screen_f.close()