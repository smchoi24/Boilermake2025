import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()


# Function to calculate the angle between three points
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

    # print(spine_angle)

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


# Initialize OpenCV video capture
cap = cv2.VideoCapture(0)  # Use 0 for the default camera

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
            cv2.putText(frame, "Unsafe Pose Detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                        cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Real-Time Pose Detection', frame)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()