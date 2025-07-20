#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ZED2i Depth and Confidence Image Verification Script

This script extracts and analyzes depth and confidence images from a ROS bag file 
recorded from a ZED2i stereo camera. It provides diagnostic information about
the images and saves them as PNG files for visual inspection.
"""

import rosbag
import cv2
import numpy as np
from cv_bridge import CvBridge
import os
import glob
import getpass

# Initialize the CV bridge for converting between ROS image messages and OpenCV images
bridge = CvBridge()

user_name = getpass.getuser()
print("Running as user:", user_name)
# Directory to search for the bag file
bag_dir = f"/home/{user_name}/grand_tour_dataset/"
if not os.path.isdir(bag_dir):
    print(f"Warning: Directory '{bag_dir}' does not exist.")
    exit(1)

# Search for files ending with _jetson_zed2i_depth.bag
bag_files = glob.glob(os.path.join(bag_dir, "*_jetson_zed2i_depth.bag"))
if not bag_files:
    print(f"Warning: No bag file ending with '_jetson_zed2i_depth.bag' found in directory: {bag_dir}")
    print("Please follow the instructions in [0]_Accessing_GrandTour_Data.ipynb to obtain the required ROS bag files before running this script.")
    exit(1)

    
# Use the first matching file (or handle as needed)
bag_file_path = os.path.abspath(bag_files[0])
print("Using bag file:", bag_file_path)

# Open the ROS bag file containing ZED2i camera data
bag_file = rosbag.Bag(bag_file_path)

# Flags to track whether we've processed each image type
is_confidence_reached = False
is_depth_reached = False

# Iterate through messages in the bag file, filtering for depth and confidence image topics
for topic, ros_msg, timestamp in bag_file.read_messages(topics=["/boxi/zed2i/depth/image_raw/compressedDepth", 
                                                  "/boxi/zed2i/confidence/image_raw/compressedDepth"]):

    # Process depth image
    if "depth" in topic:
        # Convert compressed ROS image message to OpenCV format
        depth_image = bridge.compressed_imgmsg_to_cv2(ros_msg, desired_encoding="passthrough")
        
        # Print diagnostic information about the depth image
        print("Image info:")
        print("  Type:", type(depth_image))
        print("  Dtype:", depth_image.dtype)
        print("  Shape:", depth_image.shape)
        print("  Min value:", np.min(depth_image))
        print("  Max value:", np.max(depth_image))
        print("  Unique values:", np.unique(depth_image)[:10], "... (showing first 10)")
        print("  Nonzero count:", np.count_nonzero(depth_image))
        print("  NaN count:", np.isnan(depth_image).sum() if np.issubdtype(depth_image.dtype, np.floating) else 0)
        print("  Inf count:", np.isinf(depth_image).sum() if np.issubdtype(depth_image.dtype, np.floating) else 0)

        # Additional information
        print("min:", depth_image.min())
        print("max:", depth_image.max())
        print("format:", ros_msg.format)
        
        # Sample random pixel values for analysis
        flat_depth_image = depth_image.flatten()
        depth_sample_indices = np.random.choice(flat_depth_image.size, 100, replace=False)
        print("random 100 pixel values:", flat_depth_image[depth_sample_indices])
        
        # Save the depth image for visual inspection
        cv2.imwrite("depth_image.png", depth_image)
        is_depth_reached = True

    # Process confidence image
    if "confidence" in topic:
        # Convert compressed ROS image message to OpenCV format
        confidence_image = bridge.compressed_imgmsg_to_cv2(ros_msg, desired_encoding="passthrough")
        
        # Print diagnostic information about the confidence image
        print("Confidence Image info:")
        print("  Type:", type(confidence_image))
        print("  Dtype:", confidence_image.dtype)
        print("  Shape:", confidence_image.shape)
        print("  Min value:", np.min(confidence_image))
        print("  Max value:", np.max(confidence_image))
        print("  Unique values:", np.unique(confidence_image)[:10], "... (showing first 10)")
        print("  Nonzero count:", np.count_nonzero(confidence_image))
        print("  NaN count:", np.isnan(confidence_image).sum() if np.issubdtype(confidence_image.dtype, np.floating) else 0)
        print("  Inf count:", np.isinf(confidence_image).sum() if np.issubdtype(confidence_image.dtype, np.floating) else 0)

        # Additional information
        print("min:", confidence_image.min())
        print("max:", confidence_image.max())
        print("format:", ros_msg.format)
        
        # Sample random pixel values for analysis
        flat_confidence_image = confidence_image.flatten()
        confidence_sample_indices = np.random.choice(flat_confidence_image.size, 100, replace=False)
        print("random 100 pixel values:", flat_confidence_image[confidence_sample_indices])
        
        # Save the confidence image for visual inspection
        cv2.imwrite("confidence_image.png", confidence_image)
        is_confidence_reached = True
    
    # Exit the loop once we've processed both image types
    if is_depth_reached and is_confidence_reached:
        print("Both depth and confidence images processed successfully.")
        break