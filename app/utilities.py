import os
import numpy as np
import cv2
from glob import glob
from PIL import ImageColor


def save_frame(video_path, save_dir):
    save_path = os.path.join(save_dir, "frame")

    cap = cv2.VideoCapture(video_path)
    idx = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            cap.release()
            break

        if idx == 0:
            cv2.imwrite(f"{save_path}.png", frame)

        idx += 1


## COURT MAPPING ##
def create_kalman_filter():
    kf = cv2.KalmanFilter(4, 2)
    kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
    kf.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
    kf.processNoiseCov = 1e-5 * np.eye(4, dtype=np.float32)
    kf.measurementNoiseCov = 1e-1 * np.eye(2, dtype=np.float32)
    kf.errorCovPost = 1e-1 * np.eye(4, dtype=np.float32)
    return kf


def find_homography_matrix(real_points, court_points):
    real_points = np.array(real_points, dtype=np.float32)
    court_points = np.array(court_points, dtype=np.float32)

    # Compute the homography matrix
    H, status = cv2.findHomography(real_points, court_points, cv2.RANSAC)
    return H


def get_jersey_color(position, frame, teams):
    x1, y1, x2, y2 = position
    w, h = x2 - x1, y2 - y1
    player = frame[int(y1 + (h / 5)):int(y2 - (h / 2)), int(x1 + (w / 5)):int(x2 - (w / 5))]
    player = cv2.cvtColor(player, cv2.COLOR_BGR2RGB)
    cv2.imshow('player', player)

    ratios = []
    for (lower, upper) in teams:
        lower = np.array(ImageColor.getrgb(lower))
        upper = np.array(ImageColor.getrgb(upper))
        mask = cv2.inRange(player, lower, upper)
        output = cv2.bitwise_and(player, player, mask=mask)

        ratios.append(round(np.count_nonzero(mask) / np.size(mask) * 100, 1))

        # fig, axs = plt.subplots(1, 3, figsize=(30, 15))
        # axs[0].imshow(output)
        # axs[1].imshow(player)
        # axs[2].imshow([[lower], [upper]])
        # plt.show()

    if ratios[0] > ratios[1]:
        return list(ImageColor.getrgb(teams[0][1]))
    else:
        return list(ImageColor.getrgb(teams[1][1]))


def transform(position, homography_matrix):
    x1, y1, x2, y2 = position
    old_loc = np.array((float((x1 + x2 + 1) / 2), float(y2)))  # x, y of feet
    old_loc = old_loc[np.newaxis][np.newaxis]
    new_pos = cv2.perspectiveTransform(np.array(old_loc, dtype="float32"), homography_matrix)
    new_pos = new_pos.flatten()

    return [int(i) for i in new_pos]

class Mapping:
    def __init__(self, first_frame, court_img):
        ### User point selection ###
        # Define images
        self.selection_img = None
        self.court_img = court_img
        self.first_frame = first_frame
        self.display_frame = first_frame.copy()
        self.count = 0
        # Arrays to be filled with selected points
        self.points_selected = []
        self.representation = []
        # Point selection function
        self.select_points()

        ### Setup Kalman filters for each chosen point ###
        self.prev_frame = first_frame.copy()
        self.prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

        self.points_selected = np.array(self.points_selected, dtype=np.float32)
        self.kalman_filters = [create_kalman_filter() for _ in self.points_selected]
        self.lk_params = dict(winSize=(15, 15), maxLevel=2,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        for kf, point in zip(self.kalman_filters, self.points_selected):
            kf.statePost = np.array([[point[0]], [point[1]], [0], [0]], np.float32)

    def select_points(self):
        # Displays frame and court to allow user to choose related points
        height = self.court_img.shape[0]
        width = int(self.display_frame.shape[1] * height / self.display_frame.shape[0])
        self.display_frame = cv2.resize(self.display_frame, (width, height))
        self.selection_img = np.hstack((self.display_frame, self.court_img))
        cv2.imshow('image', self.selection_img)

        # setting mouse handler for the image
        # and calling the click_event() function
        cv2.setMouseCallback('image', self.click_event)

        # Close selection
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # function to display the coordinates of
    # the points clicked on the image
    def click_event(self, event, x, y, flags, params):
        # checking for left mouse clicks
        if event == cv2.EVENT_LBUTTONDOWN:
            # displaying the coordinates
            # on the Shell
            print(x, ' ', y)
            height = self.court_img.shape[0]
            width = self.display_frame.shape[1]
            if self.count % 2:
                self.representation.append((x - width, y))
            else:
                og_height, og_width = self.first_frame.shape[0], self.first_frame.shape[1]
                height_ratio, width_ratio = og_height / height, og_width / width
                print((width_ratio, height_ratio))
                self.points_selected.append((x*width_ratio, y*height_ratio))

            # displaying the coordinates
            # on the image window
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(self.selection_img, str(self.count // 2 + 1), (x, y), font,
                        4, (255, 255, 255), 6)
            cv2.circle(self.selection_img, (x, y),
                       8, (0, 0, 0), -1)
            cv2.imshow('image', self.selection_img)
            self.count += 1

    def map(self, frame):
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Calculate optical flow to track the points
        new_points, status, error = cv2.calcOpticalFlowPyrLK(self.prev_gray, frame_gray, self.points_selected,
                                                             None, **self.lk_params)
        status = status.squeeze()
        # Update Kalman Filters and points
        print(self.kalman_filters)
        for i, kf in enumerate(self.kalman_filters):
            # Kalman prediction
            predicted = kf.predict()

            if status[i] == 1:
                # Optical flow found the point
                measurement = np.array([[new_points[i, 0]], [new_points[i, 1]]], np.float32)

                # Use Kalman prediction if the deviation is too large (possible occlusion)
                # print(np.linalg.norm(measurement - predicted[:2]))
                if np.linalg.norm(measurement - predicted[:2]) > 3:
                    self.points_selected[i] = predicted[:2].T
                else:
                    kf.correct(measurement)
                    self.points_selected[i] = measurement.T
            else:
                # Point was lost, use the Kalman prediction
                self.points_selected[i] = predicted[:2].T

        self.prev_gray = frame_gray.copy()
        H = find_homography_matrix(self.points_selected, self.representation)
        return H, self.points_selected


class Tracking:
    def __init__(self):
        self.lol = 1

