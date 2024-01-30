import cv2
import numpy as np
import matplotlib.pyplot as plt

# Assuming these functions are defined in 'util.py'
from util import get_parking_spots_bboxes, empty_or_not

# Function to calculate the absolute difference between two images
def calc_diff(im1, im2):
    return np.abs(np.mean(im1) - np.mean(im2))

# Path to the mask and video file
mask_path = './mask_1920_1080.png'
video_path = './sample/parking_1920_1080_loop.mp4'

# Read the mask image in grayscale
mask = cv2.imread(mask_path, 0)

# Open the video file
cap = cv2.VideoCapture(video_path)

# Find connected components in the mask
connected_components = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)

# Get bounding boxes for parking spots
spots = get_parking_spots_bboxes(connected_components)

# Initialize lists for spots' status and differences
spots_status = [None for _ in spots]
diffs = [None for _ in spots]

# Initialize variable for storing the previous frame
previous_frame = None

# Print the bounding box of the first spot
print(spots[0])

# Set the frame processing step and initialize frame counter
step = 30
frame_nmr = 0

# Main loop for video processing
while True:
    # Read a frame from the video
    ret, frame = cap.read()

    # Check if the frame should be processed
    if frame_nmr % step == 0 and previous_frame is not None:
        # Calculate differences for each parking spot
        for spot_index, spot in enumerate(spots):
            x1, y1, w, h = spot
            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
            diffs[spot_index] = calc_diff(spot_crop, previous_frame[y1:y1 + h, x1:x1 + w, :])
        print([diffs[j] for j in np.argsort(diffs)][::-1])

    # Process the frame every 'step' frames
    if frame_nmr % step == 0:
        for spot_index, spot in enumerate(spots):
            x1, y1, w, h = spot
            spot_crop = frame[y1:y1 + h, x1:x1 + w, :]
            spot_status = empty_or_not(spot_crop)
            spots_status[spot_index] = spot_status

    # Save the current frame for the next iteration
    if frame_nmr % step == 0:
        previous_frame = frame.copy()

    # Draw rectangles based on parking spots' status
    for spot_index, spot in enumerate(spots):
        spot_status = spots_status[spot_index]
        x1, y1, w, h = spots[spot_index]
        if spot_status:
            frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
        else:
            frame = cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 0, 255), 2)

    # Display the number of available spots
    cv2.rectangle(frame, (80, 20), (550, 80), (128, 0, 0), -1)
    cv2.putText(frame, 'Available Spots: {}/{}'.format(str(sum(spots_status)), str(len(spots))),
                (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Show the frame
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.imshow('frame', frame)

    # Break the loop if 'q' key is pressed
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    # Increment frame counter
    frame_nmr += 1

# Release video capture and close all windows
cap.release()
cv2.destroyAllWindows()
