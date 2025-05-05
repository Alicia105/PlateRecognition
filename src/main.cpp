#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <getopt.h>

#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include "../include/inference.h"

//#include <tesseract/baseapi.h>
//#include <leptonica/allheaders.h>

/*tesseract::TessBaseAPI *ocr = new tesseract::TessBaseAPI();
ocr->Init(NULL, "eng"); // English language
ocr->SetImage(image.data, image.cols, image.rows, 3, image.step);
std::string outText = std::string(ocr->GetUTF8Text());*/

using namespace std;

struct TrackableObject {
    cv::Ptr<cv::TrackerCSRT> tracker;
    int id;
    cv::Rect2d box;
    string className;
    cv::Scalar color;
    bool toRemove = false;
};

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
                        newObj.tracker = cv::TrackerCSRT::create();
                        newObj.tracker->init(resizedFrame, detection.box);
                        newObj.id =id++;
                        newObj.box = detection.box;
                        newObj.className = detection.className;
                        newObj.color = detection.color;
                        activeObjects.push_back(newObj);
                        continue;
                    }

                    for (auto& obj : activeObjects) {
                        bool ok = obj.tracker->update(resizedFrame, obj.box);
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
                            newObj.tracker = cv::TrackerCSRT::create();
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
    cout<<"Number of images ="<<num<<" , "<<"Treated ="<<treated<<" , "<<"Dropped ="<<dropped<<endl;

    return 0;
}

