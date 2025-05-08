#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>
#include "../include/inference.h"
#include "../include/sort.h"

//#include <getopt.h>
//#include <tesseract/baseapi.h>
//#include <leptonica/allheaders.h>

/*tesseract::TessBaseAPI *ocr = new tesseract::TessBaseAPI();
ocr->Init(NULL, "eng"); // English language
ocr->SetImage(image.data, image.cols, image.rows, 3, image.step);
std::string outText = std::string(ocr->GetUTF8Text());*/

using namespace std;
using namespace sort;

double computeIoU(const cv::Rect& boxA, const cv::Rect& boxB) {
    int xA = std::max(boxA.x, boxB.x);
    int yA = std::max(boxA.y, boxB.y);
    int xB = std::min(boxA.x + boxA.width, boxB.x + boxB.width);
    int yB = std::min(boxA.y + boxA.height, boxB.y + boxB.height);

    int interArea = std::max(0, xB - xA) * std::max(0, yB - yA);
    int boxAArea = boxA.width * boxA.height;
    int boxBArea = boxB.width * boxB.height;

    return (double)interArea / (boxAArea + boxBArea - interArea);
}

/*string runOCR(cv::Mat plate) {
    tesseract::TessBaseAPI tess;
    tess.Init(NULL, "eng", tesseract::OEM_LSTM_ONLY);
    tess.SetPageSegMode(tesseract::PSM_SINGLE_BLOCK);
    
    tess.SetImage(plate.data, plate.cols, plate.rows, 1, plate.step);
    std::string outText = tess.GetUTF8Text();
    
    return outText;
}*/

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
    string classNameFilePath="../models/coco.names";
    string classPlateFilePath="../models/plate.names"; 

    vector<string> allClasses=getClassNames( classNameFilePath);
    vector<string> wantedClasses={"person","bicycle","car","motorbike","bus","truck"}; 
    //vector<TrackableObject> activeObjects;
    vector<cv::Scalar> savedColors;
    
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

                // Detection box
                cv::rectangle(resizedFrame, plateBox, plateColor, 2);
                        
                // Detection box text
                string classStringPlate = plateDetection.className + ' ' + to_string(plateDetection.confidence).substr(0, 4);
                cv::Size textSizePlate = cv::getTextSize(classStringPlate, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
                cv::Rect textBoxPlate(plateBox.x, plateBox.y - 40, textSizePlate.width + 10, textSizePlate.height + 20);

                cv::rectangle(resizedFrame, textBoxPlate, plateColor, cv::FILLED);
                cv::putText(resizedFrame, classStringPlate, cv::Point(plateBox.x + 5, plateBox.y - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);

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

    bool runOnGPU = false;
    string pathToVideo="../videos/2103099-uhd_3840_2160_30fps.mp4";
    string pathToCarModel="../models/yolov8n.onnx";
    string pathToPlateModel="../models/yolov8n_plate.onnx";
    string classNameFilePath="../models/coco.names";
    string classPlateFilePath="../models/plate.names"; 
    vector<string> wantedClasses={"person","bicycle","car","motorbike","bus","truck"};  
    
    cv::VideoCapture cap(pathToVideo);

    if (!cap.isOpened()) {
        cerr << "Error: Could not open video stream.\n";
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
    int num=0;

    while (cap.isOpened()) {

        cv::Mat frame,resizedFrame;
        cap >> frame;
        cv::resize(frame,resizedFrame,cv::Size(resized_width,resized_height));

        if (frame.empty()) break;
        
        vector<Detection> output = inf.runInference(resizedFrame);
        int detections = output.size();
        cout << "Number of detections:" << detections <<endl;

        float plateConfidenceThreshold=0.5;

        for(int i=0; i<detections; ++i){
            Detection detection = output[i];

            for(auto name : wantedClasses){
                if (name==detection.className){

                    cv::Rect box = detection.box;
                    cv::Scalar color = detection.color;

                    // Detection box
                    cv::rectangle(resizedFrame, box, color, 2);

                    // Detection box text
                    string classString = detection.className + ' ' + to_string(detection.confidence).substr(0, 4);
                    cv::Size textSize = cv::getTextSize(classString, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
                    cv::Rect textBox(box.x, box.y - 40, textSize.width + 10, textSize.height + 20);

                    cv::rectangle(resizedFrame, textBox, color, cv::FILLED);
                    cv::putText(resizedFrame, classString, cv::Point(box.x + 5, box.y - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);

                    //Add tracking here
                    //Add plate recognition
                        cv::Mat imagePlate=resizedFrame(box);
                        vector<Detection> outputPlate = infPlate.runInference(imagePlate);
                        int plates=outputPlate.size();

                        for(int j=0; j<plates; ++j){
                            Detection plateDetection=outputPlate[j];
                            if(plateDetection.confidence<plateConfidenceThreshold) continue;

                            cv::Rect plateBox = plateDetection.box;
                            cv::Scalar plateColor = color;

                            //Tesseract OCR
                            /*cv::Mat plateROI = img(plateBox).clone();
                            cv::Mat gray, thresh;
                            cv::cvtColor(plateROI, gray, cv::COLOR_BGR2GRAY);
                            cv::threshold(gray, thresh, 0, 255, cv::THRESH_BINARY + cv::THRESH_OTSU);
                            string plateText = runOCR(thresh);
                            cout << "Detected plate text: " << plateText << endl;

                            //Get coordinates in original picture
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
            }     
                
            treated++;
            
        }

        cv::imshow("Original",resizedFrame);
        num++;
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
    cout<<"Number of images ="<<num<<" , "<<"Treated ="<<treated<<" , "<<"Dropped ="<<dropped<<endl;

    return 0;
}*/