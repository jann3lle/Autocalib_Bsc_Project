import os
import cv2

# record video live from webcam and save image when 's' is pressed
def record_video(save_folder, save_all_frames=False):
    # create save folder if it doesn't exist
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    # open webcam
    cap = cv2.VideoCapture(1)
    # check if webcam is opened
    if not cap.isOpened():
        print("Error opening video stream or file")
    # Read until video is completed
    while cap.isOpened():

        # Capture frame-by-frame
        ret, frame = cap.read()

        if ret:
            # Display the resulting frame
            cv2.imshow('Frame', frame)

            
            # if user presses 'q' break loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            if save_all_frames:
                # get save path- save_folder/00000X.png
                save_pth = f'{save_folder}/{len(os.listdir(save_folder)):05d}.png'
                cv2.imwrite(f'{save_pth}', frame)
                #continue 
            # save image when 's' is pressed
            #if cv2.waitKey(1) & 0xFF == ord('s'):
                # get save path- save_folder/00000X.png
                #save_pth = f'{save_folder}/{len(os.listdir(save_folder)):05d}.png'
                #cv2.imwrite(f'{save_pth}', frame)
        
    # When everything done, release the video capture object
    cap.release()
    # Closes all the frames
    cv2.destroyAllWindows()

#def continue_function():

    #for i in range(10):
        #if i == 5:
            #continue
        #print(i)

if __name__ == '__main__':

    #continue_function()
    record_video('data/calibration/z5_frames', save_all_frames=True)