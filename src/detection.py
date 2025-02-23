import time
import cv2
import mediapipe as mp
from keras.src.backend import switch
from roboflow import Roboflow
import threading
import concurrent.futures


# Roboflow Model (for vest, hard hat, and person)
rf = Roboflow(api_key="88iBd1SmUZCSTcdn6y2l")
project = rf.workspace().project("construction-safety-gsnvb")
rf_model = project.version(1).model


# Mediapipe Model (for goggles)
model_path = '../exported_model/model.tflite'

BaseOptions = mp.tasks.BaseOptions
DetectionResult = mp.tasks.components.containers.DetectionResult
ObjectDetector = mp.tasks.vision.ObjectDetector
ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

frame_count = 0

# Roboflow Preparation

last_person_detected = False
last_helmet_detected = False
last_vest_detected = False

latest_response = None
response_lock = threading.Lock()
inference_running = False

latest_detections = []

def run_inference(frame):
    """Run the model prediction on the given frame in a separate thread."""
    if frame_count % 3 == 0:
        global latest_response, inference_running
        # Run prediction on the frame (this call is blocking)
        prediction = rf_model.predict(frame, confidence=40, overlap=50)
        response = prediction.json()
        with response_lock:
            latest_response = response
        inference_running = False

executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

# Mediapipe Preparation

def print_result(result: DetectionResult, output_image: mp.Image, timestamp_ms: int):
    print('detection result: {}'.format(result))
    global latest_detections
    # Update the detections from the asynchronous callback.
    latest_detections = result.detections

options = ObjectDetectorOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    max_results=5,
    result_callback=print_result)

cap = cv2.VideoCapture(0)


# LOOP START

with ObjectDetector.create_from_options(options) as detector:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        frame_count += 1

        ##### Roboflow (hat/vest/people) #####
        inference_running = True
        frame_copy = frame.copy()
        executor.submit(run_inference, frame_copy)

        with response_lock:
            current_response = latest_response


        ##### Mediapipe (goggles) #####
        numpy_frame_from_opencv = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_timestamp_ms = int(time.time() * 1000)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_frame_from_opencv)
        if frame_count % 2 == 0:
            detector.detect_async(mp_image, frame_timestamp_ms)

        frame_height, frame_width, _ = frame.shape


        # Rectangle for helmet & vest
        if current_response is not None:
            frame_height, frame_width, _ = frame.shape
            for pred in current_response["predictions"]:
                obj_class = pred["class"]
                if obj_class in ["person", "helmet", "vest"]:
                    w = int(pred["width"])
                    h = int(pred["height"])
                    x = int(pred["x"] - (w / 2))
                    y = int(pred["y"] - (h / 2))

                    match obj_class:
                        case "person":
                            box_color = (0, 255, 0)
                        case "helmet":
                            box_color = (255, 0, 0)
                        case "vest":
                            box_color = (207, 255, 4)
                        case _:
                            box_color = (0, 0, 0)

                    print(f"Detected {obj_class} with coordinates: x={x}, y={y}, width={w}, height={h}")

                    cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 4)
                    label = f"{obj_class}: {pred['confidence']:.2f}"
                    cv2.putText(frame, label, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

        # Rectangle for goggles
        for detection in latest_detections:
            if detection.categories[0].score > 0.5:
                rbox = detection.bounding_box
                width = int(rbox.width)
                height = int(rbox.height)
                x = int(rbox.origin_x)
                y = int(rbox.origin_y)

                cv2.rectangle(frame, (x, y), (x+width, y+height), (255, 123, 0), 2)

                label = f"{detection.categories[0].category_name}:{detection.categories[0].score:.2f}"
                cv2.putText(frame, label, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 123, 0), 2)

        cv2.imshow("Safety Features", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
executor.shutdown()