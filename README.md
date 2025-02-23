# Site Sentinel

Construction Site Safety Monitoring Software

## Description

Site Sentinel is a computer vision-based safety monitoring system designed specifically for construction sites. It uses advanced visual object detection and pose estimation to ensure that workers comply with safety standards by wearing the correct personal protective equipment (PPE) and using proper lifting techniques.

## Table of Contents
* Features
* Technologies Used
* Getting Started
  * Prerequisites
  * Installation
  * Configuration
* Usage
* Project Structure
* Contributing
* Contact

## Features

* Real-Time Object Detection:
  * Uses the Roboflow API along with a custom-trained Mediapipe object detection model to detect hard hats, safety vests, and goggles from webcam footage.

* Pose Detection for Safety:
  * Employs Mediapipe’s pose detection model to analyze a worker's posture during heavy lifting, sending warnings if unsafe postures are detected.

* Live Video Processing:
  * Captures computer visuals via the OpenCV library, processes each frame in real time, and overlays detection annotations on the screen.

* Safety Code Enforcement:
  * Automatically checks if detected objects and postures meet safety standards, alerting users to any violations.

## Technologies Used
* OpenCV: Captures and processes live video streams from a webcam.
* Roboflow API: Integrates with custom object detection models for PPE identification.
* Mediapipe Object Detection Model: Custom model to detect safety equipments such as gogggles.
* Mediapipe Pose Detection Model: Analyzes worker posture, especially during heavy lifting.
* Mediapipe's Model Maker Library: Used for training and fine-tuning detection models to recognize certain objects.

## Getting Started

### Prerequisites

* Python 3.11
* A functioning webcam
* Libraries
  * OpenCV
  * mediapipe
  * roboflow
  * Other dependencies as required

### Configuration

* Roboflow API Key:
Set up your API_KEY environment variables through config.py

```
API_KEY = "your_api_key"
```

## Usage

Run the ppe application to start monitoring ppe objects:

```
python ppe_detection.py
```

Or the pose application to start monitoring poses of workers:

```
python pose_detection.py
```

Upon execution, Site Sentinel will:

* Access the webcam to capture live video.
* Process each frame using OpenCV.
* Detect hard hats, safety vests, and goggles using the Roboflow API and custom object detection model.
* Evaluate worker posture via Mediapipe’s pose detection.
* Display annotated video feeds and issue warnings when safety violations are detected.

## Project structure

* **src/:** contains all source code
  * **ppe_detection.py:** source code for PPE detection program.
  * **pose_detection.py:** source code for pose detection program.
  * **exported_model/:** contains custom trained model from mediapipe.
* **docs/:** contains all frontend related assets and code
  * **assets/:** contains css, js, and needed assets.
  * **images/:** contains essential images for website frontend.
  * **templates:** contains HTML files of web pages.

## Authors
 
- [Peter Yuk](https://github.com/dyuk01)
- [Sehyeong Oh](https://github.com/Sehonp05)
- [Steve Choi](https://github.com/smchoi24)
- [Seongjae Park](https://github.com/spright786)

## Version History

* 0.1
    * Initial Release

## Contributing

Contributions are welcome! Feel free to fork the repository and submit pull requests. For major changes, please open an issue first to discuss your ideas.

## Contact

For questions, suggestions, or issues, please open an issue on the GitHub repository or contact the maintainer at peteryuk91@gmail.com.


