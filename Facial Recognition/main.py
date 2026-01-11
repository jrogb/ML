import cv2 
import mediapipe as mp
import os
import argparse

# Create output directory if it doesn't exist to save results
output_dir = './data/output'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
img_path = './data/testImg.jpg'

# Initialize Mediapipe Face Detection
mp_face_detection = mp.solutions.face_detection

# Function to process image and blur detected faces or draw rectangles around detected faced
def process_img(img, face_detection):
    H, W, _ = img.shape # Get image dimensions
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert BGR to RGB
    result = face_detection.process(img_rgb) # Perform face detection
    
    # Draw rectangles around detected faces or blur them
    if result.detections is not None: # If faces are detected
        for result in result.detections:# Iterate through detected faces
            location_data = result.location_data # Get location data
            bbox = location_data.relative_bounding_box # Get bounding box

            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height # Get bbox coordinates
            padding_x = 0.05 # Add padding to bbox
            padding_y = 0.1

            x1 = int((bbox.xmin - padding_x) * W) # Calculate top-left x coordinate with padding
            y1 = int((bbox.ymin - padding_y) * H) # Calculate top-left y coordinate with padding
            w = int((bbox.width + 2*padding_x) * W) # Calculate width with padding
            h = int((bbox.height + 2*padding_y) * H) # Calculate height with padding

            extra_top = int(0.2 * h) # Extra top padding for better coverage
            x1 = max(x1, 0) # Ensure x1 is within image bounds
            y1 = max(y1-extra_top, 0) # Ensure y1 is within image bounds
            w = min(w, W - x1) # Ensure width is within image bounds
            h = min(h + extra_top, H - y1) # Ensure height is within image bounds

            img = cv2.rectangle(img, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2) # Draw rectangle around face
            #img[y1:y1 + h, x1:x1 + w] = cv2.GaussianBlur(img[y1:y1 + h, x1:x1 + w], (99, 99), 0) # Blur the face region

    return img


# Alternative function to blur faces that is more effective/memory efficient
"""def process_img(img, face_detection):
    H, W, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = face_detection.process(img_rgb)
    
    if result.detections is not None:
        for detection in result.detections:
            location_data = detection.location_data
            bbox = location_data.relative_bounding_box

            x1, y1, w, h = bbox.xmin, bbox.ymin, bbox.width, bbox.height
            x1 = int(x1 * W)
            y1 = int(y1 * H)
            w = int(w * W)
            h = int(h * H)

            # Extract the face ROI
            face_roi = img[y1:y1+h, x1:x1+w]

            # Downscale the ROI to make blurring cheaper but stronger
            small = cv2.resize(face_roi, (0,0), fx=0.3, fy=0.3)

            # Apply Gaussian blur on the smaller image
            blurred = cv2.GaussianBlur(small, (99,99), 0)

            # Upscale back to original ROI size
            face_roi_blurred = cv2.resize(blurred, (w, h))

            # Replace the original ROI with the blurred one
            img[y1:y1+h, x1:x1+w] = face_roi_blurred

    return img"""

args = argparse.ArgumentParser() # Argument parser for command line inputs
args.add_argument('--mode', default='webcam') # Mode: image, video, webcam
args.add_argument('--filePath', default='./data/video.mp4') # File path for image or video
args = args.parse_args() # Parse the arguments

# Initialize face detection model
with mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection: # Use context manager to ensure proper resource management

    if args.mode in ['image']: # Image mode
        img_original = cv2.imread(args.filePath) # Read input image
        img = cv2.resize(img_original, (640, 480)) # Resize image for consistent processing
    
        img = process_img(img, face_detection) # Process the image to blur faces

        cv2.imwrite(os.path.join(output_dir, 'blurred_face.jpg'), img) # Save the processed image

    elif args.mode in ['video']: # Video mode
        cap = cv2.VideoCapture(args.filePath) # Read input video
        ret, frame = cap.read() # Read the first frame

        output_video = cv2.VideoWriter(os.path.join(output_dir, 'blurred_face_video2.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 25.0, (frame.shape[1], frame.shape[0])) # Initialize video writer to save output video
        
        # Process each frame of the video
        while ret:
            frame = process_img(frame, face_detection) # Process the frame to blur faces
            output_video.write(frame) # Write the processed frame to output video
            ret, frame = cap.read() # Read the next frame

        cap.release() # Release video capture object
        output_video.release() # Release video writer object
    
    elif args.mode in ['webcam']: # Webcam mode
        cap = cv2.VideoCapture(0) # Open webcam
        ret, frame = cap.read() # Read the first frame

        # Process each frame from the webcam
        while ret:
            frame = process_img(frame, face_detection) # Process the frame to blur faces
            cv2.imshow('Webcam - Blurred Faces', frame) # Display the processed frame
            if cv2.waitKey(25) & 0xFF == ord('q'): # Exit on 'q' key press
                break

            ret, frame = cap.read() # Read the next frame

        cap.release() # Release webcam
        cv2.destroyAllWindows() # Close all OpenCV windows
