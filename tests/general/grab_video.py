import os
import cv2

def record_video(save_folder, video_filename='output.mp4', fps=20.0, frame_width=640, frame_height=480):
    # Create save folder if it doesn't exist
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
    save_path = os.path.join(save_folder, video_filename)
    out = cv2.VideoWriter(save_path, fourcc, fps, (frame_width, frame_height))
    
    # Open webcam
    cap = cv2.VideoCapture(1)
    
    if not cap.isOpened():
        print("Error opening video stream or file")
        return
    
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # Resize frame (optional)
            frame = cv2.resize(frame, (frame_width, frame_height))
            
            # Write the frame to the video file
            out.write(frame)
            
            # Display the resulting frame
            cv2.imshow('Recording - Press Q to stop', frame)
            
            # Stop recording when 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    
    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Video saved at {save_path}")

if __name__ == '__main__':
    record_video('data/calibration/zchange_video', 'zchange_video.mp4')
