#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String, Float32MultiArray
import cv2
import numpy as np
import pyrealsense2 as rs
from button_detection import ButtonDetector
from character_recognition import CharacterRecognizer

class ButtonDetectionNode(Node):
    def __init__(self):
        super().__init__('button_detection_node')

        # Publishers
        self.button_positions_pub = self.create_publisher(Float32MultiArray, 'button_positions', 10)
        self.button_text_pub = self.create_publisher(String, 'button_text', 10)

        # Initialize camera and detector
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)

        # Initialize the button detector and recognizer
        self.detector = ButtonDetector()
        self.recognizer = CharacterRecognizer()

        # Timer for processing frames at 30Hz
        self.timer = self.create_timer(1.0/30.0, self.process_frame)

    def process_frame(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            return

        img_np = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Detect buttons in the frame
        boxes, scores, _ = self.detector.predict(img_np, True)
        button_patches, button_positions, _ = button_candidates(boxes, scores, img_np)

        # Detect 3D coordinates of buttons
        button_3d_coordinates = get_button_3d_with_normal(button_positions, depth_frame, color_frame.profile.as_video_stream_profile().intrinsics)

        # Publish detected buttons' 3D positions
        if button_3d_coordinates:
            button_positions_msg = Float32MultiArray()
            button_positions_msg.data = [coord for position in button_3d_coordinates for coord in position]
            self.button_positions_pub.publish(button_positions_msg)

        # Recognize and publish button text
        for button_img in button_patches:
            button_text, _, _ = self.recognizer.predict(button_img)
            self.button_text_pub.publish(String(data=button_text))

    def destroy(self):
        self.pipeline.stop()
        super().destroy()

def main(args=None):
    rclpy.init(args=args)

    button_detection_node = ButtonDetectionNode()

    try:
        rclpy.spin(button_detection_node)
    except KeyboardInterrupt:
        pass

    button_detection_node.destroy()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
