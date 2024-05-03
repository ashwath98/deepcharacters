import cv2
import os
import sys
camera_list=[i for i in range(116)] #list of cameras
camera_number=int(sys.argv[1])# camera number
camera_number=camera_list[camera_number]
subject_name=sys.argv[2]#'Subject0003'
type=sys.argv[3]#'tight'
split=sys.argv[4]#'testing'
subject_myfolder_name=sys.argv[5]#'VladNew'
base_folder='/scratch/inf0/user/asshetty/dummy_data/'+subject_myfolder_name+'/'+split+'/images'#path where you want to dump the images
video_path = '/CT/ashwath2/static00/DatasetRelease/'+subject_name+'/'+split+'/videos'# path where you have the videos unzipped
start_frame= sys.argv[6]
end_frame = sys.argv[7]
print(start_frame)
print(end_frame)
print(base_folder)
print(video_path)
if not os.path.isdir(base_folder):
    os.mkdir(base_folder)
output_folder=os.path.join(base_folder,str(camera_number))
if not os.path.exists(output_folder):
    os.mkdir(output_folder)
# Open the video file

video_path =os.path.join(video_path,'stream'+str(camera_number).zfill(3)+'.mp4')

# Open the video filesss
print(video_path)
video = cv2.VideoCapture(video_path)
print(video)
# Get the total number of frames in the video
total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
print(total_frames) # edit here for lower number of frames
if end_frame==-1:
    end_frame=total_frames
# Define the exact frame numbers you want to extract
frame_numbers = [i for i in range(start_frame,end_frame)]

# Loop through each frame in the video

for i in range(total_frames):
    if i not in frame_numbers:
        continue
    if i%1000==0:
        print(i)
    # Check if the current frame number is in the list of desired frame numbers
   
        
        # Read the current frame
    ret, frame = video.read()
    if not ret:
        break
        # Do something with the frame (e.g. save it to a file)
    cv2.imwrite(os.path.join(output_folder,f"image_c_{camera_number}_f_{i}.jpg"), frame)
        
# Release the video object
video.release()
