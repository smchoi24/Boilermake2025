from flask import Flask, render_template, Response
import cv2
import time
import mediapipe as mp
from roboflow import Roboflow
import threading
import concurrent.futures
from config import API_KEY

app = Flask(__name__)

################################
# Roboflow Initialization
################################
rf = Roboflow(api_key=API_KEY)
project = rf.workspace().project("construction-safety-gsnvb")
rf_model = project.version(1).model

################################
# Mediapipe Initialization
################################
model_path = '../src/exported_model/model.tflite'  # <-- adjust this path as needed

BaseOptions = mp.tasks.BaseOptions
DetectionResult = mp.tasks.components.containers.DetectionResult
ObjectDetector = mp.tasks.vision.ObjectDetector
ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# We'll hold the latest Mediapipe detections here
latest_detections = []

def print_result(result: DetectionResult, output_image: mp.Image, timestamp_ms: int):
    """
    This callback will be triggered by Mediapipe after each detect_async call.
    We store the latest detections so we can draw them on the frame in real-time.
    """
    global latest_detections
    latest_detections = result.detections

options = ObjectDetectorOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.LIVE_STREAM,
    max_results=5,
    result_callback=print_result
)

################################
# Thread Pool (for Roboflow)
################################
executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

################################
# Capture Device
################################
cap = cv2.VideoCapture(0)  # or replace with the path to a video file if desired
frame_count = 0

# Thread safety for Roboflow predictions
latest_response = None
response_lock = threading.Lock()
inference_running = False

def run_inference(frame):
    """
    Submit the frame to Roboflow.
    We store the JSON response in a global variable so it can be drawn later.
    """
    global latest_response, inference_running

    # Run prediction on the frame (blocking call)
    prediction = rf_model.predict(frame, confidence=40, overlap=50)
    response = prediction.json()

    with response_lock:
        latest_response = response
    inference_running = False


################################
# Main Streaming Generator
################################
def generate_frames():
    """
    Generates video frames (with bounding boxes drawn) in MJPEG format for browser streaming.
    """
    global frame_count, inference_running

    # Instantiate Mediapipe object detector once outside the loop
    with ObjectDetector.create_from_options(options) as detector:
        while True:
            success, frame = cap.read()
            if not success:
                break  # no more frames, or camera not available

            frame_count += 1

            #######################
            # Roboflow inference
            #######################
            # For performance, maybe run inference every N frames (e.g., every 3 frames).
            if frame_count % 3 == 0 and not inference_running:
                inference_running = True
                frame_copy = frame.copy()
                executor.submit(run_inference, frame_copy)

            # Get current response safely
            with response_lock:
                current_response = latest_response

            #######################
            # Mediapipe inference
            #######################
            # Convert from BGR to RGB for Mediapipe
            numpy_frame_from_opencv = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_timestamp_ms = int(time.time() * 1000)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=numpy_frame_from_opencv)

            # If you want Mediapipe to run at a different rate, do so here
            if frame_count % 2 == 0:
                detector.detect_async(mp_image, frame_timestamp_ms)

            #######################
            # Drawing bounding boxes
            #######################
            # 1) Roboflow bounding boxes (person, vest, helmet)
            if current_response is not None:
                for pred in current_response.get("predictions", []):
                    obj_class = pred["class"]
                    if obj_class in ["person", "helmet", "vest"]:
                        w = int(pred["width"])
                        h = int(pred["height"])
                        x = int(pred["x"] - w / 2)
                        y = int(pred["y"] - h / 2)

                        if obj_class == "person":
                            box_color = (0, 255, 0)
                        elif obj_class == "helmet":
                            box_color = (255, 0, 0)
                        elif obj_class == "vest":
                            box_color = (207, 255, 4)
                        else:
                            box_color = (0, 0, 0)

                        cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 4)
                        label = f"{obj_class}: {pred['confidence']:.2f}"
                        cv2.putText(frame, label, (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

            # 2) Mediapipe bounding boxes (goggles)
            for detection in latest_detections:
                # If the confidence is above your threshold
                if detection.categories[0].score > 0.5:
                    rbox = detection.bounding_box
                    width = int(rbox.width)
                    height = int(rbox.height)
                    x = int(rbox.origin_x)
                    y = int(rbox.origin_y)

                    cv2.rectangle(frame, (x, y), (x + width, y + height), (255, 123, 0), 2)
                    label = f"{detection.categories[0].category_name}:{detection.categories[0].score:.2f}"
                    cv2.putText(frame, label, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 123, 0), 2)

            #######################
            # Encode frame for MJPEG
            #######################
            ret, buffer = cv2.imencode('.jpg', frame)
            # Create an HTTP Multipart response
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    # Cleanup
    cap.release()
    executor.shutdown()


################################
# Flask Routes
################################
@app.route('/')
def index():
    """
    Render the main page with an <img> tag that will receive the video stream.
    """
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """
    Route dedicated to providing the MJPEG stream.
    """
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


################################
# Run the Flask app
################################
if __name__ == '__main__':
    # Start the stream as soon as the app starts
    threading.Thread(target=generate_frames, daemon=True).start()

    # Run Flask server
    app.run(host='0.0.0.0', port=5000, debug=False)
