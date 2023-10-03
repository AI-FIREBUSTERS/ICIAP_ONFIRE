import cv2, os, argparse
import numpy as np
import os

# Import custom modules
from utility import Utils
from result_evaluator import ResultEvaluator

def init_parameter():   
     # Set up a parser to handle command-line arguments
    parser = argparse.ArgumentParser(description='Test')

    # Define two command-line arguments: the locations of the video dataset and the results folder
    parser.add_argument("--videos", type=str, default='foo_videos/', help="Dataset folder")
    parser.add_argument("--results", type=str, default='foo_results/', help="Results folder")
    
    # Parse the arguments and return them
    args = parser.parse_args()
    return args

# Parse the command-line arguments
args = init_parameter()

# Here you should initialize your method

# Setting verbose to False, to control the amount of output printed to the console
verbose = False

# Load the pre-trained neural network model
PATH_WEIGHTS = "best.pt"
MODEL_CONFIDENCE = 0.35
model = Utils.load_net(PATH_WEIGHTS, MODEL_CONFIDENCE)
stride, pt = model.stride, model.pt # model stride and padding

# Set the width and height of the frames that the model will process
WIDTH = 640
HEIGHT = 640
imgsz = (WIDTH,HEIGHT)

# Check if the image size is compatible with the model stride, and adjust if necessary
imgsz = Utils.check_img_size(imgsz, s=stride)  # check img_size

# Initialize the ResultEvaluator, which will analyze the output of the model
CONFIDENCE_THRESHOLD = 0.4
result_evaluator = ResultEvaluator(verbose=False, confidence_threshold=0.4)

# If the fire is detected consecutively for TOTAL_FIRE_SEC seconds, the fire is detected
TOTAL_FIRE_SEC = 0.5

################################################

print("Starting the test...")
# Start the main loop, which will process each video
for video in os.listdir(args.videos):

    ret = True

    # Open the video file
    cap = cv2.VideoCapture(os.path.join(args.videos, video))

    # Get the frames per second (fps) of the video
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Initialize a flag to track whether this is the first frame of a new video
    new_video = True

    # Initialize a counter for the number of frames processed in the current video
    frame_video_counter = 0

    if verbose:
        print("Processing video ", video, " with FPS: ", fps)

    # Start the loop to process each frame of the video
    while ret:
        # Here you should add your code for applying your method 

        # Read the next frame from the video file
        ret, img = cap.read()

        # If a frame was successfully read
        if ret:
            # Prepare the frame for input to the model
            img1 = Utils.letterbox(img, imgsz, stride=stride, auto=pt)[0]  # padded resize
            img1 = img1.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
            img1 = np.ascontiguousarray(img1)  # contiguous

            # If this is the first frame of a new video
            if new_video:
                # Set the frame dimensions in the ResultEvaluator
                result_evaluator.set_frame_dimensions(img1.shape[1], img1.shape[2])
                # Reset the ResultEvaluator for the new video
                result_evaluator.reset()
                # Reset the flag for the next video
                new_video = False
                # Sets the fire threshold in order to detect a fire if it is present in TOTAL_FIRE_SEC seconds
                result_evaluator.set_fire_threshold(int(fps*TOTAL_FIRE_SEC))
            
            frame_video_counter += 1

            # Get the model's predictions for this frame
            results = model(img1)
            # Evaluate the results
            result_evaluator.evaluate(results, frame_video_counter)
            
            # If a fire has been detected
            if result_evaluator.detected:
                # If verbose mode is enabled, print a message
                if verbose:
                    print("Fire detected after {} frame in the video {}, this means fire after {} seconds".format(frame_video_counter,video, frame_video_counter/fps))
                # Stop processing this video
                ret = False

    ########################################################

    # Release the video file
    cap.release()
    # Here you should add your code for writing the results

    # Write the result (time of fire detection in seconds) to a text file
    if result_evaluator.detected:
        with open(args.results+str(video).split('.')[0]+".txt", "w") as f:
            f.write(str(round(frame_video_counter/fps)))
    else:
        # If no fire was detected, write an empty string to the file
        with open(args.results+str(video).split('.')[0]+".txt", "w") as f:
            f.write("")
    ########################################################

print("Test completed.")