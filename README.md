# PlateRecognition

## **Description**  
A C++ implementation of a real-time plate recognition system. This project process traffic videos to detect vehicules and their license plate.

## **Table of Contents**     
- [Features](#features)
- [Tech Stack](#teck-stack)
- [Installation](#installation) 
- [How to use](#how-to-use)
- [Demo](#demo)
- [Acknowledgement](#acknowledgement)


## ⚙️**Features**
- Real-time object detection with Yolov8 model
- Multi-object tracking with SORT algorithm
- License plate detection with customed Yolov8 model
- Optical Character Recognition (OCR) to get plate characters with Fast-OCR model
- Save license plate and vehicules timestamp in a SQLite database

## 🛠️ **Tech Stack**
- **C++ 11 or higher**
- **OpenCV 4.11.0 or higher** 
- **A C++ compiler** 
- **SQLite C amalgamation**  

## 🧪**Installation**  
- Clone or download the repository
<pre> git clone https://github.com/Alicia105/PlateRecognition.git </pre>
- Navigate to the project directory
<pre> cd PlateRecognition </pre>

## 🚀**How to use** 
1. Navigate to the project directory
<pre> cd PlateRecognition </pre>

2. Create a build directory
<pre> mkdir build
 cd build </pre>

3. Run the CMake
<pre> cmake ..</pre>

4. Build the project with make
<pre> make </pre>

5. Run the executable file
<pre> ./PlateRecognition </pre>

## **Demo**

https://github.com/user-attachments/assets/962b7fff-bc7b-4678-ad0c-bfdcb7c9b6b3

## **Acknowledgment** 
- SORT algorithm : https://github.com/itsuki2021/sort-cpp
- Fast Plate OCR : https://github.com/ankandrew/fast-plate-ocr/releases/tag/v0.3.0
- License plate dataset (by @computervisioneng) : https://universe.roboflow.com/roboflow-universe-projects/license-plate-recognition-rxg4e/dataset/4
