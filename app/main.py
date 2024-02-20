from collections import defaultdict

import cvzone
from ultralytics import YOLO

from utilities import transform, get_jersey_color
from utilities import Mapping

from sort import *

import matplotlib
import math
import cv2

matplotlib.use("QT5Agg", force=True)

def visualise(video_path, court, teams):
    # Initialise Video
    cap = cv2.VideoCapture(video_path)
    _, prev_frame = cap.read()
    first_frame = cv2.imread('../data/frame.png')

    # Initialise Mapping
    mapping = Mapping(first_frame, court)

    # Load Detection Model
    model = YOLO('../models/best.pt')

    # Tracking
    track_history = defaultdict(lambda: [])
    players = defaultdict(list)  # {id: position history}
    references = {}  # New ids assignment to existing players, {new id: existing id}
    ass = False

    # Loop through the video frames
    count = 0
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()
        if success:
            ######################### COURT MAPPING ############################
            H, points_selected = mapping.map(frame)

            ######################### PLAYER DETECTION #########################
            # Run object detection on the frame, persisting tracks between frames
            results = model.track(frame, persist=True)
            for r in results:
                # Initialise Images
                annotated_frame = r.plot()
                annotated_court = court.copy()
                annotated_court = cv2.cvtColor(annotated_court, cv2.COLOR_BGR2RGB)
                temp_detections = defaultdict(list)
                # Draw the tracks (for visualization)
                for i, new in enumerate(points_selected):
                    a, b = new.ravel()
                    a, b = int(a), int(b)
                    cv2.circle(annotated_frame, (a, b), 5, color=(0, 0, 255), thickness=-1)
                    a, b = transform((a, b, a, b), H)
                    cv2.circle(annotated_court, (a, b), 5, color=(255, 0, 255), thickness=-1)
                player_class = {1: "Player"}
                for box in r.boxes:
                    box_class = int(box.cls[0])
                    conf = math.ceil(box.conf[0] * 100) / 100
                    if box_class in list(player_class.keys()) and conf > 0.5:
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        player = (x1, y1, x2, y2)

                        jersey_color = get_jersey_color(player, frame, teams)
                        new_loc = transform(player, H)
                        cv2.circle(annotated_court, new_loc, 25, get_jersey_color, -1)

                        ## TRACKING ##
                        box_id = int(box.id[0])
                        temp_detections[box_id] = [(x1 + x2) / 2, (y1 + y2) / 2]

                        if box_id in references:
                            cvzone.putTextRect(annotated_court,
                                               f'ID: {references[box_id]}',
                                               new_loc, scale=5, thickness=1, colorR=(0, 0, 0))

                        # cv2.rectangle(ann, new_loc, [i + 5 for i in new_loc], 123)
                        track = track_history[box_id]
                        track.append(new_loc)  # x, y center point
                        if len(track) > 30:  # retain 90 tracks for 90 frames
                            track.pop(0)
                        points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                        cv2.polylines(annotated_court, [points], isClosed=False, color=(0, 0, 0), thickness=10)

                if len(temp_detections) == 10 and not ass:
                    players = temp_detections
                    for i in players:
                        references[i] = i
                    ass = True
                if ass:
                    not_assigned = set()
                    assigned = set()
                    for i in temp_detections:
                        if i in players:
                            players[i].append(temp_detections[i])
                            if len(players[i]) > 30:  # retain 90 tracks for 90 frames
                                players[i].pop(0)
                            assigned.add(i)
                        elif i in references:
                            players[references[i]].append(temp_detections[i])
                            assigned.add(references[i])
                        else:
                            not_assigned.add(i)

                    for u in not_assigned:
                        least = np.inf
                        pos = 0
                        for i in (set(players.keys()) - assigned):
                            if np.linalg.norm(np.array(temp_detections[u]) - np.array(players[i][-1])) < least:
                                least = np.linalg.norm(np.array(temp_detections[u]) - np.array(players[i][-1]))
                                pos = i
                        references[u] = pos
                        players[pos].append(temp_detections[u])

                height = annotated_court.shape[0]  # Height of annotated_court
                width = int(annotated_frame.shape[1] * height / annotated_frame.shape[0])
                annotated_frame = cv2.resize(annotated_frame, (width, height))
                annotated_court = cv2.cvtColor(annotated_court, cv2.COLOR_BGR2RGB)
                combined_image = np.hstack((annotated_frame, annotated_court))

                # Display the combined image using OpenCV
                cv2.imshow("Court and Frame", combined_image)

                # Wait for a short period before the next frame
                # 20 ms should give a good frame rate
                if cv2.waitKey(2) & 0xFF == ord('q'):
                    break

        else:
            # Break the loop if the end of the video is reached
            break
        count += 1


if __name__ == '__main__':
    videos = ["../data/play.mp4"]

    empty_court = "../data/orl.jpeg"
    empty_court = cv2.imread(empty_court)

    team_colors = [('#7C0503', '#B84153'), ('#211559', '#3E47A6')]

    visualise(videos[0], empty_court, team_colors)
