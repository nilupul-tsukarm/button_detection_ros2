#!/usr/bin/env python
from __future__ import print_function
import os
import cv2
import numpy as np
import pyrealsense2 as rs
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from button_detection import ButtonDetector
from character_recognition import CharacterRecognizer
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray

class ButtonPublisher(Node):
    def __init__(self):
        super().__init__('button_publisher')
        self.publisher = self.create_publisher(MarkerArray, 'button_markers', 10)

    def publish_button_markers(self, button_3d_coordinates, button_positions):
        marker_array = MarkerArray()
        
        for i, (button_3d, button_pos) in enumerate(zip(button_3d_coordinates, button_positions)):
            x, y, z = button_3d
            
            # Create a marker for the button center
            marker = Marker()
            marker.header.frame_id = "camera_frame"
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "buttons"
            marker.id = i
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = x
            marker.pose.position.y = y
            marker.pose.position.z = z
            marker.scale.x = 0.05
            marker.scale.y = 0.05
            marker.scale.z = 0.05
            marker.color.a = 1.0
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0  # Green for button centers

            marker_array.markers.append(marker)

            # Create a bounding box around the button
            x_min, y_min, x_max, y_max = button_pos
            box_marker = Marker()
            box_marker.header.frame_id = "camera_frame"
            box_marker.header.stamp = self.get_clock().now().to_msg()
            box_marker.ns = "button_boxes"
            box_marker.id = i + 1000  # Unique ID for bounding box
            box_marker.type = Marker.CUBE
            box_marker.action = Marker.ADD
            box_marker.pose.position.x = x
            box_marker.pose.position.y = y
            box_marker.pose.position.z = z
            box_marker.scale.x = (x_max - x_min) * 0.001  # Scale for bounding box
            box_marker.scale.y = (y_max - y_min) * 0.001
            box_marker.scale.z = 0.01  # Small thickness for the box
            box_marker.color.a = 0.5  # Semi-transparent box
            box_marker.color.r = 0.0
            box_marker.color.g = 0.0
            box_marker.color.b = 1.0  # Blue for bounding boxes
            
            marker_array.markers.append(box_marker)

        # Publish marker array
        self.publisher.publish(marker_array)

def process_realsense_video(detector, recognizer, button_publisher):
    # Configure RealSense camera
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    # Start the camera
    pipeline.start(config)
    align = rs.align(rs.stream.color)

    # Get camera intrinsics
    profile = pipeline.get_active_profile()
    color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
    intrinsics = color_stream.get_intrinsics()

    try:
        normal_vector = None  # Initialize normal_vector to None
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            img_np = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            
            # Detect buttons in the frame
            boxes, scores, _ = detector.predict(img_np, True)
            button_patches, button_positions, _ = button_candidates(boxes, scores, img_np)

            # Detect plane and get normal vector
            _, normal_vector = detect_plane_and_get_normal(depth_image, intrinsics)

            if normal_vector is not None:
                # Get 3D coordinates and plot normal axes
                button_3d_coordinates = get_button_3d_with_normal(button_positions, depth_frame, intrinsics, normal_vector, img_np)
                
                # Publish button markers
                button_publisher.publish_button_markers(button_3d_coordinates, button_positions)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    button_publisher = ButtonPublisher()

    detector = ButtonDetector()
    recognizer = CharacterRecognizer()

    detector.predict(np.zeros((480, 640, 3), dtype=np.uint8), True)
    recognizer.predict(np.zeros((180, 180, 3), dtype=np.uint8))

    process_realsense_video(detector, recognizer, button_publisher)

    rclpy.spin(button_publisher)
    button_publisher.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
