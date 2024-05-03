import cv2
import os
import sys
camera_list=[i for i in range(116)]
camera_number=int(sys.argv[1])
camera_number=camera_list[camera_number]
subject_name=sys.argv[2]#'Subject0028'
type=sys.argv[3]#'free_talking'
split=sys.argv[4]#'training'
subject_myfolder_name=sys.argv[5]#'Heming'
downsampled=sys.argv[6]

video_path = '/CT/ashwath2/static00/DatasetRelease/'+subject_name+'/'+split+'/segmentation_videos/'+'stream'+str(camera_number).zfill(3)+'.mp4'


base_folder='/scratch/inf0/user/asshetty/dummy_data/'+subject_myfolder_name+'/'+split+'/foregroundSegmentation'

start_frame= sys.argv[7]
end_frame = sys.argv[8]
print(start_frame)
print(end_frame)
print(base_folder)
print(video_path)
print(base_folder)
print(video_path)
if not os.path.isdir(base_folder):
    os.mkdir(base_folder)
output_folder=os.path.join(base_folder,str(camera_number))
if not os.path.exists(output_folder):
    os.mkdir(output_folder)
# Open the video file


# Open the video file
video = cv2.VideoCapture(video_path)
if end_frame==-1:
    end_frame=total_frames
# Get the total number of frames in the video m usujdj
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
print(total_frames)
# Define the exact frame numbers you want to extract
frame_numbers = [i for i in range(start_frame,end_frame)]

# Loop through each frame in the video
for i in range(total_frames):
    if i%1000==0:
        print(i)
    # Check if the current frame number is in the list of desired frame numbers
    if i in frame_numbers:
        # Set the current frame to the desired frame number
        video.set(cv2.CAP_PROP_POS_FRAMES, i)
        
        # Read the current frame
        ret, frame = video.read()
        
        # Do something with the frame (e.g. save it to a file)
        cv2.imwrite(os.path.join(output_folder,f"image_c_{camera_number}_f_{i}.jpg"),  frame)
        
# Release the video object
video.release()
