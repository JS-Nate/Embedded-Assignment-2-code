
import cv2
import os

def process_frame(frame):
    """
    Process a single frame (for example, perform image segmentation or detection).
    You can add any custom processing logic here.
    
    :param frame: The image frame to process.
    """
    # Example: You can apply transformations, detections, etc. on the frame
    # Currently, we will just return the frame as is
    return frame

def process_video(input_video_path, output_video_path):
    """
    Reads a video file, processes each frame, and writes the results to an output video.
    
    :param input_video_path: Path to the input video file.
    :param output_video_path: Path to save the output video.
    """
    # Open the input video
    cap = cv2.VideoCapture(input_video_path)
    
    # Get video frame properties
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Create a VideoWriter object to save the processed video using 'XVID' codec
    out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (frame_width, frame_height))





    frame_count = 0
    max_frames = 300  # Stop after processing 300 frames

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # End of video
        
        if frame_count >= max_frames:
            break  # Stop after processing 300 frames

        # Process the frame (e.g., detection, transformation)
        processed_frame = process_frame(frame)

        # Write the processed frame to the output video
        out.write(processed_frame)

        frame_count += 1

    # Release the video capture and writer objects
    cap.release()
    out.release()
    print(f"Processed {frame_count} frames and saved to {output_video_path}")

# Example usage
if __name__ == "__main__":
    input_video_path = 'video.mp4.f247.webm'  # Input video path in the current directory
    output_video_path = 'output_video.mp4'  # Output video path in the same directory
    process_video(input_video_path, output_video_path)

