import cv2
import os
from pathlib import Path
import sys

def create_video_from_images(folder1, folder2, output_video_path, fps=15):
    # Get list of image files in both folders
    images1 = sorted(Path(folder1).glob("*.png"))
    images2 = sorted(Path(folder2).glob("*.png"))
    
    # Ensure both folders have the same number of images
    assert len(images1) == len(images2), "Both folders must have the same number of images"
    
    # Read the first image to get the dimensions
    first_image1 = cv2.imread(str(images1[0]))
    first_image2 = cv2.imread(str(images2[0]))
    
    # Ensure both images have the same width
    assert first_image1.shape[1] == first_image2.shape[1], "Images must have the same width"
    
    # Get the width and height for the video
    height = first_image1.shape[0] + first_image2.shape[0]
    width = first_image1.shape[1]
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    
    for img1_path, img2_path in zip(images1, images2):
        # Read images
        img1 = cv2.imread(str(img1_path))
        img2 = cv2.imread(str(img2_path))
        
        # Concatenate images vertically
        concatenated_image = cv2.vconcat([img1, img2])
        
        # Write the frame to the video
        out.write(concatenated_image)
    
    # Release the video writer
    out.release()
    print(f"Video saved to {output_video_path}")

# Example usage
folder1 = sys.argv[1]
folder2 = sys.argv[2]
output_video_path = sys.argv[3]
create_video_from_images(folder1, folder2, output_video_path)

