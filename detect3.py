import cv2

# Load the pre-trained Haar Cascade classifier for vehicle detection
haar_cascade = 'car.xml'  # Replace with the actual path to your Haar Cascade XML
input_video = 'video.mp4.f247.webm'  # Path to your input video
output_video = 'output_video.avi'  # Path to save the output video

# Create a VideoCapture object to read the input video
cap = cv2.VideoCapture(input_video)
car_cascade = cv2.CascadeClassifier(haar_cascade)

# Get the video frame width, height, and frames per second (FPS) to set the output video parameters
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Create VideoWriter object to write the output video
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can change the codec if needed
out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

# Process the first 300 frames
frame_count = 0
while True:
    ret, frames = cap.read()
    if not ret:
        break

    # Stop processing after 300 frames
    if frame_count >= 300:
        break

    # Convert the frame to grayscale for vehicle detection
    gray = cv2.cvtColor(frames, cv2.COLOR_BGR2GRAY)

    # Detect vehicles (cars) in the frame
    cars = car_cascade.detectMultiScale(gray, 1.1, 1)

    # Draw rectangles around the detected vehicles
    for (x, y, w, h) in cars:
        cv2.rectangle(frames, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Write the processed frame to the output video
    out.write(frames)

    # Display the frame with detected vehicles
    cv2.imshow('Vehicle Detection', frames)

    # Increment frame count
    frame_count += 1

    # Wait for 33ms and check if the user presses the 'Esc' key to exit
    if cv2.waitKey(33) == 27:  # 27 is the ASCII code for the 'Esc' key
        break

# Release video capture and writer objects and close all windows
cap.release()
out.release()
cv2.destroyAllWindows()

