import cv2

# Load the pre-trained Haar Cascade classifier for vehicle detection
haar_cascade = 'car.xml'  # Replace with the actual path to your Haar Cascade XML
input_video = 'video.mp4.f247.webm'  # Path to your input video
output_video = 'output_video.avi'  # Path to save the output video

# Create a VideoCapture object to read the input video
cap = cv2.VideoCapture(input_video)
car_cascade = cv2.CascadeClassifier(haar_cascade)

# Check if the input video and Haar Cascade loaded successfully
if not cap.isOpened():
    print("Error: Could not open input video.")
    exit()
if car_cascade.empty():
    print("Error: Could not load Haar Cascade XML.")
    exit()

# Get the video frame width, height, and frames per second (FPS) to set the output video parameters
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Create VideoWriter object to write the output video
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for AVI format
out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

# Process the first 300 frames
frame_count = 0
while True:
    ret, frames = cap.read()
    if not ret:
        print("End of video or error reading frame.")
        break

    # Stop processing after 300 frames
    if frame_count >= 300:
        print("Processed 300 frames. Stopping.")
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

    # Increment frame count
    frame_count += 1

# Release video capture and writer objects
cap.release()
out.release()

print(f"Output video saved as '{output_video}'.")


