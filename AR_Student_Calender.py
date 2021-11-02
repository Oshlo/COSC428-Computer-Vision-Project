import cv2
import cv2.aruco as aruco
import numpy as np
from datetime import datetime


def get_lecture_image(day, hour):

    # Hard Coded Calender Data :(

    file = "Calander/Blank.png"

    if day == 0:
        if 9 <= hour < 11:
            file = "Calander/ENME303Lecture.png"
            # ENME303 lecture
        elif 11 <= hour < 12:
            file = "Calander/ENEL373Lecture.png"
            # ENCE373 lecture
        elif 12 <= hour < 14:
            file = "Calander/COSC428Lab.png"
            # COSC428 lab
    elif day == 1:
        if 10 <= hour < 11:
            file = "Calander/ENCE461Lecture.png"
            # ENCE461 lecture
        elif 15 <= hour < 17:
            file = "Calander/ENEL373Lab.png"
            # ENEL373 Lab
    elif day == 2:
        if 9 <= hour < 10:
            file = "Calander/ENME303Lecture.png"
            # ENME303 lecture
        elif 10 <= hour < 11:
            file = "Calander/ENEL373Lecture.png"
            # ENEL373 lecture
        elif 11 <= hour < 13:
            file = "Calander/ENCE461Lab.png"
            # ENCE461 lab
        elif 13 <= hour < 14:
            file = "Calander/ENME303Tutorial.png"
            # ENME303 tutorial
        elif 16 <= hour < 17:
            file = "Calander/ENEL373Tutorial.png"
            # ENEL373 tutorial
    elif day == 3:
        if 9 <= hour < 11:
            file = "Calander/COSC428Lecture.png"
            # COSC lecture
        elif 13 <= hour < 14:
            file = "Calander/ENCE461Lecture.png"
            # ENCE461 lecture
    elif day == 4:
        if 9 <= hour < 10:
            file = "Calander/ENEL373Lecture.png"
            # ENEL373 lecture
        if 10 <= hour < 11:
            file = "Calander/ENME303Lecture.png"
            # ENME303 lecture

    return file


def find_aruco_markers(img, camera_matrix, dist_coeff, marker_size=6, total_markers=250, draw_axis=True,
                       draw_bounding_box=True):
    # Set webcam image to greyscale
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Get the dictionary for the ArUco markers based on the input parameters
    key = getattr(aruco, f'DICT_{marker_size}X{marker_size}_{total_markers}')
    aruco_dict = aruco.Dictionary_get(key)
    aruco_param = aruco.DetectorParameters_create()

    # Get the ArUco bounding boxes and IDs
    bounding_boxs, ids, _ = aruco.detectMarkers(img_grey, aruco_dict, parameters=aruco_param)

    # Draw an axis on each of the detected ArUco markers
    if draw_axis and ids:
        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(bounding_boxs, 0.05, camera_matrix, dist_coeff)
        print(rvecs, tvecs)
        for i in range(0, len(ids)):
            aruco.drawAxis(img, camera_matrix, dist_coeff, rvecs[i], tvecs[i], 0.05)

    # Draw a bounding box on each of the detected ArUco markers
    if draw_bounding_box:
        aruco.drawDetectedMarkers(img, bounding_boxs)

    return [bounding_boxs, ids]


def augment_aruco(bounding_box, id, img, img_augment, draw_id=True, multiplier=2):

    # Get the corners of the ArUco bounding box
    tl = bounding_box[0][0][0], bounding_box[0][0][1]
    tr = bounding_box[0][1][0], bounding_box[0][1][1]
    br = bounding_box[0][2][0], bounding_box[0][2][1]
    bl = bounding_box[0][3][0], bounding_box[0][3][1]

    # Find the center of the ArUco marker, used for transforming the matrix
    c = ((tl[0]+tr[0]+br[0]+bl[0])/4), ((tl[1]+tr[1]+br[1]+bl[1])/4)
    c = np.array(c, dtype=int)

    # Apply transformation so calender images aren't fitted to the ArUco bounding box
    tl = tl[0] + multiplier*(tl[0] - c[0]), tl[1] + multiplier*(tl[1] - c[1]) - 100
    tr = tr[0] + multiplier*(tr[0] - c[0]), tr[1] + multiplier*(tr[1] - c[1]) - 100
    br = br[0] + multiplier*(br[0] - c[0]), br[1] + multiplier*(br[1] - c[1]) - 100
    bl = bl[0] + multiplier*(bl[0] - c[0]), bl[1] + multiplier*(bl[1] - c[1]) - 100

    # Get the calender image height and width for transformation
    h, w, _ = img_augment.shape

    # # Draw a circle at the centre
    # img = cv2.circle(img, (c[0], c[1]), radius=2, color=(0, 0, 255), thickness=-1)

    # Array of the Bounding box corners
    pts1 = np.array([tl, tr, br, bl])

    # Array of image corners for transformation
    pts2 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

    # Use the above arrays to find the homography of the image
    matrix, _ = cv2.findHomography(pts2, pts1)

    # apply the matrix to the calender image
    img_out = cv2.warpPerspective(img_augment, matrix, (img.shape[1], img.shape[0]))

    # combine the new transformed calender image and the webcam image
    cv2.fillConvexPoly(img, pts1.astype(int), (0, 0, 0))
    img_out = img + img_out

    return img_out


def augment_markerless(img, img_disp, img_target, kp1, kp2, good):
    # Find the data for the bounding box from the good key points
    src_points = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_points = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # Find homography
    matrix, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5)

    h, w, _ = img_target.shape

    # Same as for ArUco markers
    pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
    dst = cv2.perspectiveTransform(pts, matrix)

    # Same as for ArUco markers
    img_warp = cv2.warpPerspective(img_disp, matrix, (img.shape[1], img.shape[0]))
    cv2.fillConvexPoly(img, dst.astype(int), (0, 0, 0))
    img = img + img_warp

    return img


def main():

    # # Get real time data
    # now = datetime.now()
    # hour = int(now.strftime("%H"))  # str
    # day = datetime.today().weekday()  # int

    # Spoof time data
    hour = 16
    day = 2

    # Set video capture from built-in webcam
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(3, 1920)
    cap.set(4, 1080)

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

    # Select the calender file to load
    file = get_lecture_image(day, hour)
    img_aug = cv2.imread(file)
    img_aug = cv2.flip(img_aug, 1) # will be re-flipped

    # Load pre made camera matrix and distortion coefficients
    camera_matrix = np.loadtxt('Camera/camera_matrix.npy')
    distortion_coeff = np.loadtxt('Camera/distortion_coeff.npy')

    # Get the target image for markerless ar
    img_target = cv2.imread("Images/ID2.jpg")
    hT, wT, _ = img_target.shape
    img_disp = cv2.resize(img_aug, (wT, hT))

    # use ORB to create mapped key points from a reference image, to look for in the webcam image
    orb = cv2.ORB_create(nfeatures=1500)
    key_points_1, descriptor_1 = orb.detectAndCompute(img_target, None)
    img_target = cv2.drawKeypoints(img_target, key_points_1, None)

    while True:
        # read image
        success, img = cap.read()

        # Markerless Stuff:

        # Get the key points from the webcam image
        key_points_2, descriptor_2 = orb.detectAndCompute(img, None)

        # Use a brute force algorithm to detect matching key points
        brute_force = cv2.BFMatcher()
        matches = brute_force.knnMatch(descriptor_1, descriptor_2, k=2)
        good = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good.append(m)

        # Check if enough markers match, then find the bounding box, and map the calender images to the webcam image
        if len(good) > 35:
            img = augment_markerless(img, img_disp, img_target, key_points_1, key_points_2, good)

        # ArUco Marker Stuff:

        # look in image for ArUco markers
        aruco_found = find_aruco_markers(img, camera_matrix, distortion_coeff, draw_axis=False, draw_bounding_box=False)

        # loop through markers and augment each one
        if len(aruco_found[0]) != 0:
            for bounding_box, id in zip(aruco_found[0], aruco_found[1]):
                img = augment_aruco(bounding_box, id, img, img_aug, multiplier=1)

        out.write(img)  # Write the frame to the output file.

        # Display the Image

        # Flip the output image to 'mirror' the real world
        img = cv2.flip(img, 1)
        cv2.imshow("Image", img)
        #cv2.imshow("Target Image", img_target)

        # Press ESC to close window
        if cv2.waitKey(1) & 0xFF == ord('\x1b'):  # Close the script when q is pressed.  \x1b
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
