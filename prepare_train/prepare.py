import os
import cv2

# Current folder
curdir = os.path.join(os.getcwd(), 'prepare_train')

# Prepare train folders
train_folder = os.path.join(curdir, 'dance/sequences')
os.makedirs(train_folder, exist_ok=True)

# Read videos
input_video_folder = os.path.join(curdir, 'input_videos')
list_videos = [os.path.join(input_video_folder, f)
               for f in os.listdir(input_video_folder)]

count_videos = 0
for input_vid in list_videos:
    count_videos += 1
    # Start capturing the feed
    cap = cv2.VideoCapture(input_vid)
    # Find the number of frames
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    print ("Number of frames: ", video_length)
    count = 0
    print ("Converting video..\n")
    # Start converting the video
    output_loc_ = os.path.join(train_folder, '{:05d}'.format(count_videos))
    os.makedirs(output_loc_, exist_ok=True)
    inter_video_count = 0
    to_train = False
    while cap.isOpened():
        inter_video_count += 1
        if to_train is False:
            to_train = True
        else:
            to_train = False
        # Extract three frame
        ret, frame_1 = cap.read()
        if not ret:
            break
        ret, frame_2 = cap.read()
        if not ret:
            break
        ret, frame_3 = cap.read()
        if not ret:
            break
        # Write the results back to output location.
        output_loc = os.path.join(output_loc_, '{:04d}'.format(inter_video_count))
        os.makedirs(output_loc, exist_ok=True)
        cv2.imwrite(os.path.join(output_loc, 'im1.png'), frame_1)
        cv2.imwrite(os.path.join(output_loc, 'im2.png'), frame_2)
        cv2.imwrite(os.path.join(output_loc, 'im3.png'), frame_3)
        
        if to_train is True:
            to_write = os.path.join('dance', 'tri_trainlist.txt')
        else:
            to_write = os.path.join('dance', 'tri_testlist.txt')
        with open(os.path.join(curdir, to_write), 'a') as f:
            folders = output_loc.split('/')
            f.write(os.path.join(folders[-2], folders[-1]) + '\n')
