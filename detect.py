import cv2
import datetime
import os

# Path to the input video
input_video_path = 'video.mp4.f247.webm'  # Update this with your actual video path

# Capture the input video
cap = cv2.VideoCapture(input_video_path)

# Check if the video is opened successfully
if not cap.isOpened():
    print("Error: Could not open input video.")
    exit()

# Modify the frame resolution
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

# Get the directory of the input video to save output video in the same location
output_video_path = os.path.join(os.path.dirname(input_video_path), f'output_{datetime.date.today()}.mp4')

# Create a video file in the same directory as input video
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 10, (frame_width, frame_height))

# Counter for the number of frames processed
frame_counter = 0
max_frames = 300  # Limit the processing to 300 frames

while True:
    # Capture each frame of the video
    ret, frame = cap.read()
    if ret:
        frame_counter += 1
        
        # Apply a gaussian blur to the frames
        frame = cv2.GaussianBlur(frame, (5, 5), 0)
        
        # Convert to grayscale for object detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply laplacian edge detection to convert the frames to binary format
        frame = cv2.Laplacian(src=frame, ddepth=cv2.CV_8U, ksize=3)

        # Load the pre-trained model 'car.xml' into the classifier
        car_cascade = cv2.CascadeClassifier('car.xml')  # Make sure to specify the correct path for 'car.xml'
        cars = car_cascade.detectMultiScale(gray, 1.1, 1)

        # Draw rectangles around the detected cars in each frame
        for (x, y, w, h) in cars:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Display the processed frame
        cv2.imshow("Frame", frame)

        # Write the processed frame to the output video
        out.write(frame)

        # Press 'Q' to exit early
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        
        # Stop after processing 300 frames
        if frame_counter >= max_frames:
            break
    else:
        break

# Release video capture and writer
cap.release()
out.release()

# Close all OpenCV windows
cv2.destroyAllWindows()

print(f"Processing complete. Output video saved to {output_video_path}")
