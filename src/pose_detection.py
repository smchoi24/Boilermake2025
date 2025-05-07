from dotenv import load_dotenv
import os

import cv2
import mediapipe as mp
import numpy as np
import boto3
import datetime
import time
from config import AWS_ACCESS_KEY, AWS_SECRET_KEY, S3_BUCKET_NAME, VIOLATION_COOLD  # Import config values


load_dotenv('key.env')

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Initialize S3 Client
s3_client = boto3.client(
    's3',
    region_name='us-east-2',
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY
)

# Define function to calculate spine angle
def calculate_spine_angle(shoulder, hip):
    # Convert to NumPy arrays
    shoulder = np.array(shoulder)  # Upper point
    hip = np.array(hip)  # Lower point

    # Compute the vector from hip to shoulder
    vector = shoulder - hip

    # Calculate the angle w.r.t the vertical axis (y-axis)
    angle = np.arctan2(vector[1], vector[0]) * 180.0 / np.pi

    # Convert to absolute deviation from vertical (90 degrees)
    spine_angle = abs(90 - angle)
    if spine_angle >= 180:
        spine_angle = 90 - (spine_angle - 180)
    elif spine_angle < 180:
        spine_angle -= 90

    return spine_angle

# Function to detect unsafe poses
def detect_unsafe_pose(landmarks):
    # Get landmarks
    shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
    hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]

    # Calculate angles
    spine_angle = calculate_spine_angle(shoulder, hip)

    # Define unsafe thresholds
    if spine_angle <= 60:
        return True  # Unsafe pose
    return False  # Safe pose

# Function to upload frame to S3
def upload_to_s3(frame):
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    s3_key = f"screenshots/frame{timestamp}.jpg"

    success, buffer = cv2.imencode('.jpg', frame)
    if success:
        try:
            s3_client.put_object(
                Bucket=S3_BUCKET_NAME,  # Replace with your bucket name
                Key=s3_key,
                Body=buffer.tobytes(),
                ContentType='image/jpeg'
            )
            print(f"Upload Successful: safetyviolations/{s3_key}")
        except Exception as e:
            print("Upload Failed:", e)
    else:
        print("Failed to encode frame to jpeg.")

# Initialize OpenCV video capture
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

# Initialize variables for tracking violation time
last_violation_time = None
violation_cooldown = VIOLATION_COOLD

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    # Convert the frame to RGB for MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Pose
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        # Draw pose landmarks on the frame
        mp.solutions.drawing_utils.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2,
                                                                         circle_radius=2),
            connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(color=(0, 0, 255), thickness=2)
        )

        # Detect unsafe poses
        if detect_unsafe_pose(results.pose_landmarks.landmark):
            # Check if enough time has passed  since the last violation
            current_time = time.time()
            if last_violation_time is None or current_time - last_violation_time >= violation_cooldown:
                print("Unsafe pose detected, uploading frame to S3.")
                # Upload frame to S3 if unsafe pose detected
                upload_to_s3(frame)
                last_violation_time = current_time  # Update the last violation time
        else:
            last_violation_time = None  # Reset the timer if no violation is detected

    # Display the frame
    cv2.imshow('Real-Time Pose Detection', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
