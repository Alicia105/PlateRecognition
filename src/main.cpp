#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>
#include "../include/inference.h"
#include "../include/sort.h"

using namespace std;
using namespace sort;

const string alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_";

// Preprocess the plate image to match model input
cv::Mat preprocessPlate(const cv::Mat& plate) {
    cv::Mat gray, resized;
    cv::cvtColor(plate, gray, cv::COLOR_BGR2GRAY);
    cv::resize(gray, resized, cv::Size(140, 70));
    cv::Mat blob = cv::dnn::blobFromImage(resized, 1.0, cv::Size(140, 70), cv::Scalar(), false, false, CV_8U); // NCHW
    blob=blob.reshape(1, {1, 70, 140, 1});
    cout << "Blob type: " << blob.type() <<endl;
    return blob;
}

// Decode the model output
string decodeOutput(const cv::Mat& flat_output, const string& alphabet) {
    const int num_slots = 9;
    const int num_classes = 37;
    std::string result;

    // Reshape the flat [1, 333] to [9, 37]
    cv::Mat reshaped = flat_output.reshape(1, num_slots);  // shape becomes [9, 37]

    for (int i = 0; i < num_slots; ++i) {
        cv::Mat scores = reshaped.row(i);
        cv::Point classIdPoint;
        double confidence;
        cv::minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
        int classId = classIdPoint.x;

        char c = alphabet[classId];
        if (c != '_') result += c;  // skip padding
    }
    return result;
}

string recognizePlate(cv::dnn::Net& net, const cv::Mat& plate, const string& alphabet) {
    cv::Mat inputBlob = preprocessPlate(plate);
    cout << "Blob shape: " << inputBlob.size << " Channels: " << inputBlob.channels() << endl;
    net.setInput(inputBlob);
    cv::Mat output = net.forward(); // shape [1, 9, 37], flattened to [9, 37]
   
    // Print number of dimensions
    cout << "Output dims: " << output.dims << endl;

    // Print each dimension
    cout << "Output shape: [";
    for (int i = 0; i < output.dims; ++i) {
        std::cout << output.size[i];
        if (i < output.dims - 1) cout << ", ";
    }
    std::cout << "]" << std::endl;

    return decodeOutput(output,alphabet);
}

vector<string> getClassNames(string filePath){
    vector<string> class_names;
    ifstream ifs(filePath);
    string line;
    while (getline(ifs, line)) {
        class_names.push_back(line);
    }
    cout<<"Number of classes :"<<class_names.size()<<endl;
    return class_names;
}

int main() {

    bool runOnGPU = false;
    string pathToVideo="../videos/2103099-uhd_3840_2160_30fps.mp4";
    string pathToCarModel="../models/yolov8n.onnx";
    string pathToPlateModel="../models/yolov8n_plate.onnx";
    //string pathToOcr="../models/crnn_tiny-plate.onnx";
    string pathToOcr="../models/global_mobile_vit_v2_ocr.onnx";
    string classNameFilePath="../models/coco.names";
    string classPlateFilePath="../models/plate.names"; 
    string ocrCharacterFilePath="../models/labels.txt"; 

    vector<string> allClasses=getClassNames( classNameFilePath);
    vector<string> allChar=getClassNames(ocrCharacterFilePath);
    vector<string> wantedClasses={"person","bicycle","car","motorbike","bus","truck"}; 
    vector<cv::Scalar> savedColors;

    cv::dnn::Net ocrNet = cv::dnn::readNetFromONNX(pathToOcr);
    
    cv::VideoCapture cap(pathToVideo);

    if (!cap.isOpened()) {
        cerr << "Error: Could not open video stream."<<endl;
        return -1;
    }

    Inference inf(pathToCarModel, cv::Size(640, 640), classNameFilePath, runOnGPU);
    Inference infPlate(pathToPlateModel, cv::Size(640, 640), classPlateFilePath, runOnGPU);

    float width=cap.get(cv::CAP_PROP_FRAME_WIDTH);
    float height=cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    float fps=cap.get(cv::CAP_PROP_FPS);

    int resized_width = 1280;
    int  resized_height = 720;

    int treated=0;
    int dropped=0;

    int numColors=10; //number of colors to draw boxes
    int numberOfFrame=0;

    float plateConfidenceThreshold=0.5;
    double IoUThreshold=0.5;

    Sort::Ptr sortTracker = make_shared<Sort>(1, 3, 0.3f);

    while (cap.isOpened()) {

        cv::Mat frame,resizedFrame,detectionsMat;;
        cap >> frame;

        if (frame.empty()) break;

        cv::resize(frame,resizedFrame,cv::Size(resized_width,resized_height));
        numberOfFrame++;
      
        if(numberOfFrame%5==0) continue;
        
        vector<Detection> output = inf.runInference(resizedFrame);
        int detections = output.size();
        std::cout << "Number of detections:" << detections <<endl;


        //Detection
        //input of the tracker update: detections: [xc, yc, w, h, score, class_id]
        for(int i=0; i<detections; ++i){
            Detection detection = output[i];

            for(auto name : wantedClasses){
                if (name==detection.className){

                    cv::Rect box=detection.box;
                    float center_x =box.x+box.width/2;
                    float center_y=box.y+box.height/2;

                    cv::Mat row = (cv::Mat_<float>(1,6) << center_x, center_y, box.width, box.height, detection.confidence, detection.class_id);

                    if (detectionsMat.empty()) {
                        detectionsMat = row; // first row
                    } else {
                        cv::vconcat(detectionsMat, row, detectionsMat); // append
                    }
                    
                    if (savedColors.size()<numColors){
                        savedColors.push_back(detection.color);
                    }
                    
                }
            }
        }

        //Update tracker
        cv::Mat trackedObjects = sortTracker->update(detectionsMat);
        //update output : bounding boxes estimate: [xc, yc, w, h, score, class_id, vx, vy, tracker_id]
        vector<int> validTrackerId;

        for (int j = 0; j < trackedObjects.rows; ++j) {

            float cx = trackedObjects.at<float>(j, 0);
            float cy = trackedObjects.at<float>(j, 1);
            float w  = trackedObjects.at<float>(j, 2);
            float h  = trackedObjects.at<float>(j, 3);
            float score  = trackedObjects.at<float>(j, 4);
            float class_id  = trackedObjects.at<float>(j, 5);
            float tracker_id = trackedObjects.at<float>(j, 8);

            float top_left_x=cx-w/2;
            float top_left_y=cy-h/2;

            cv::Rect bbox(top_left_x,top_left_y,w,h);

            string className=allClasses[static_cast<int>(class_id)];
            cv::Scalar color=savedColors[static_cast<int>(tracker_id)%numColors];

            // Draw Vehicules Detection box text
            cv::rectangle(resizedFrame, bbox, color, 2);

            string classString =className + ' ' + to_string(score).substr(0, 4);
            cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
            cv::Rect textBox(bbox.x, bbox.y - 40, textSize.width + 10, textSize.height + 20);

            cv::rectangle(resizedFrame, textBox, color, cv::FILLED);
            cv::putText(resizedFrame, classString, cv::Point(bbox.x + 5, bbox.y - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);

            //Plate detection
            cv::Mat imagePlate=resizedFrame(bbox);
            vector<Detection> outputPlate = infPlate.runInference(imagePlate);
            int plates=outputPlate.size();

            for(int k=0; k<plates; ++k){
                Detection plateDetection=outputPlate[k];

                if(plateDetection.confidence<plateConfidenceThreshold) continue;

                cv::Rect plateBox = plateDetection.box;
                cv::Scalar plateColor = color;
                            
                plateBox.x+=bbox.x;
                plateBox.y+=bbox.y;

                //Plate Detection box
                cv::rectangle(resizedFrame, plateBox, plateColor, 2);
                        
                //OCR in Plate Detection box
                cv::Mat plateCropped = resizedFrame(plateBox); // From plate detection
                string text = recognizePlate(ocrNet, plateCropped,alphabet);

                /*string classStringPlate = plateDetection.className + ' ' + to_string(plateDetection.confidence).substr(0, 4);
                cv::Size textSizePlate = cv::getTextSize(classStringPlate, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
                cv::Rect textBoxPlate(plateBox.x, plateBox.y - 40, textSizePlate.width + 10, textSizePlate.height + 20);*/

                //Draw plate detection + OCR 
                string classStringPlate = text;
                cv::Size textSizePlate = cv::getTextSize(classStringPlate, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
                cv::Rect textBoxPlate(plateBox.x, plateBox.y - 40, textSizePlate.width + 10, textSizePlate.height + 20);

                cv::rectangle(resizedFrame, textBoxPlate, plateColor, cv::FILLED);
                cv::putText(resizedFrame, classStringPlate, cv::Point(plateBox.x + 5, plateBox.y - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);
                cout<<text<<endl;
            }
      
            treated++;  
        }
       
        cv::imshow("Original",resizedFrame);
        int k = cv::waitKey(10); // Wait for a keystroke in the window
        if(k=='q'){break;}
    }

    cap.release();
    cv::destroyAllWindows();

    width=static_cast<int>(width);
    height=static_cast<int>(height);
    fps=static_cast<int>(fps);

    cout<<"Frame : [width="<<width<<" x height="<<height<<"]"<<endl;
    cout<<"FPS :"<<fps<<endl;
    cout<<"Number of images ="<<numberOfFrame<<" , "<<"Treated ="<<treated<<" , "<<"Dropped ="<<dropped<<endl;

    return 0;
}

/*int main() {
    cv::Mat m = cv::imread("../images/nevada.jpg");
    string ocrCharacterFilePath="../models/labels.txt";
    vector<string> allChar=getClassNames(ocrCharacterFilePath);

    if (m.empty()) {
        cerr << "Failed to load image\n";
        return -1;
    }

    string pathToOcr = "../models/global_mobile_vit_v2_ocr.onnx";
    cv::dnn::Net crnnNet = cv::dnn::readNetFromONNX(pathToOcr);
    
    vector<string> names = crnnNet.getLayerNames();

    for (const auto& name : names) cout << name <<endl;
    cout << "Input name: " << crnnNet.getUnconnectedOutLayersNames()[0] << std::endl;


    string text = recognizePlate(crnnNet,m,alphabet);
    cout << "Detected Plate: " << text << endl;

    cv::imshow("Original", m);
    int k = cv::waitKey(0);
    return 0;
}*/


/*
int main() {

    bool runOnGPU = false;
    string pathToVideo="../videos/2103099-uhd_3840_2160_30fps.mp4";
    string pathToCarModel="../models/yolov8n.onnx";
    string pathToPlateModel="../models/yolov8n_plate.onnx";
    string classNameFilePath="../models/coco.names";
    string classPlateFilePath="../models/plate.names"; 
    vector<string> wantedClasses={"person","bicycle","car","motorbike","bus","truck"}; 
    vector<TrackableObject> activeObjects;
    
    cv::VideoCapture cap(pathToVideo);

    if (!cap.isOpened()) {
        cerr << "Error: Could not open video stream."<<endl;
        return -1;
    }

    Inference inf(pathToCarModel, cv::Size(640, 640), classNameFilePath, runOnGPU);
    Inference infPlate(pathToPlateModel, cv::Size(640, 640), classPlateFilePath, runOnGPU);

    float width=cap.get(cv::CAP_PROP_FRAME_WIDTH);
    float height=cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    float fps=cap.get(cv::CAP_PROP_FPS);

    int resized_width = 1280;
    int  resized_height = 720;

    int treated=0;
    int dropped=0;

    int id=0;
    int numberOfFrame=0;

    float plateConfidenceThreshold=0.5;
    double IoUThreshold=0.5;

    while (cap.isOpened()) {

        cv::Mat frame,resizedFrame;
        cap >> frame;
        cv::resize(frame,resizedFrame,cv::Size(resized_width,resized_height));
        numberOfFrame++;

        if (frame.empty()) break;
        if(numberOfFrame%5==0) continue;
        
        vector<Detection> output = inf.runInference(resizedFrame);
        int detections = output.size();
        cout << "Number of detections:" << detections <<endl;

        //Detection
        for(int i=0; i<detections; ++i){
            Detection detection = output[i];

            for(auto name : wantedClasses){
                if (name==detection.className){
                    
                    //Tracking
                    if(activeObjects.size()==0){
                        TrackableObject newObj;
                        newObj.tracker = cv::TrackerMIL::create();
                        newObj.tracker->init(resizedFrame, detection.box);
                        newObj.id =id++;
                        newObj.box = detection.box;
                        newObj.className = detection.className;
                        newObj.color = detection.color;
                        activeObjects.push_back(newObj);
                        continue;
                    }

                    for (auto& obj : activeObjects) {
                        cv::Rect b=obj.box;
                        bool ok = obj.tracker->update(resizedFrame, b);
                        if (!ok) obj.toRemove = true;
                    }
                    
                    if (activeObjects.size()!=0) {
                        bool matched = false;
                        for (auto& obj : activeObjects) {
                            double iou = computeIoU(detection.box, obj.box);
                            if (iou > IoUThreshold) {
                                matched = true;
                                break;
                            }
                        }
                        if (!matched) {
                            TrackableObject newObj;
                            newObj.tracker = cv::TrackerMIL::create();
                            newObj.tracker->init(resizedFrame, detection.box);
                            newObj.id = id++;
                            newObj.box = detection.box;
                            newObj.className = detection.className;
                            newObj.color = detection.color;
                            activeObjects.push_back(newObj);
                        }
                    }
                    
                    // Remove lost trackers
                    activeObjects.erase(
                        remove_if(activeObjects.begin(), activeObjects.end(),
                                       [](const TrackableObject& obj) { return obj.toRemove; }),
                        activeObjects.end()
                    );

                }
            }
            
            for (auto& obj : activeObjects){
                cv::Rect box = obj.box;
                cv::Scalar color = obj.color;

                // Detection box
                cv::rectangle(resizedFrame, box, color, 2);

                // Draw Detection box text
                string classString = obj.className + ' ' + to_string(obj.confidence).substr(0, 4);
                cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
                cv::Rect textBox(box.x, box.y - 40, textSize.width + 10, textSize.height + 20);

                cv::rectangle(resizedFrame, textBox, color, cv::FILLED);
                cv::putText(resizedFrame, classString, cv::Point(box.x + 5, box.y - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);

                //Plate detection
                cv::Mat imagePlate=resizedFrame(box);
                vector<Detection> outputPlate = infPlate.runInference(imagePlate);
                int plates=outputPlate.size();

                for(int j=0; j<plates; ++j){
                    Detection plateDetection=outputPlate[j];

                    if(plateDetection.confidence<plateConfidenceThreshold) continue;

                    cv::Rect plateBox = plateDetection.box;
                    cv::Scalar plateColor = color;
                           
                    plateBox.x+=box.x;
                    plateBox.y+=box.y;

                    // Detection box
                    cv::rectangle(resizedFrame, plateBox, plateColor, 2);
                    
                    // Detection box text
                    string classStringPlate = plateDetection.className + ' ' + to_string(plateDetection.confidence).substr(0, 4);
                    cv::Size textSizePlate = cv::getTextSize(classStringPlate, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
                    cv::Rect textBoxPlate(plateBox.x, plateBox.y - 40, textSizePlate.width + 10, textSizePlate.height + 20);

                    cv::rectangle(resizedFrame, textBoxPlate, plateColor, cv::FILLED);
                    cv::putText(resizedFrame, classStringPlate, cv::Point(plateBox.x + 5, plateBox.y - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);

                }

            }
                           
            treated++;  
        }

        cv::imshow("Original",resizedFrame);
        int k = cv::waitKey(10); // Wait for a keystroke in the window
        if(k=='q'){break;}
    }

    cap.release();
    cv::destroyAllWindows();

    width=static_cast<int>(width);
    height=static_cast<int>(height);
    fps=static_cast<int>(fps);

    cout<<"Frame : [width="<<width<<" x height="<<height<<"]"<<endl;
    cout<<"FPS :"<<fps<<endl;
    cout<<"Number of images ="<<numberOfFrame<<" , "<<"Treated ="<<treated<<" , "<<"Dropped ="<<dropped<<endl;

    return 0;
}*/