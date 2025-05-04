#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <getopt.h>

#include <opencv2/opencv.hpp>
#include "../include/inference.h"

//#include <tesseract/baseapi.h>
//#include <leptonica/allheaders.h>

/*tesseract::TessBaseAPI *ocr = new tesseract::TessBaseAPI();
ocr->Init(NULL, "eng"); // English language
ocr->SetImage(image.data, image.cols, image.rows, 3, image.step);
std::string outText = std::string(ocr->GetUTF8Text());*/

using namespace std;

inline float sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

cv::Rect getUnpaddedAndScaledBox(int i, cv::Mat resizedFrame,cv::Mat output){

    float input_w = 640.0f;
    float input_h = 640.0f;

    float r_w = input_w / (float)resizedFrame.cols;  // ~0.5 for 1280
    float r_h = input_h / (float)resizedFrame.rows;  // ~0.89 for 720

    float scale = min(r_w, r_h);

    float new_unpad_w = scale * resizedFrame.cols;
    float new_unpad_h = scale * resizedFrame.rows;

    float pad_w = (input_w - new_unpad_w) / 2;
    float pad_h = (input_h - new_unpad_h) / 2;

    float pred_x = output.at<float>(i, 0);
    float pred_y = output.at<float>(i, 1);
    float pred_w = output.at<float>(i, 2);
    float pred_h = output.at<float>(i, 3);

    // YOLO outputs are relative to input_w/input_h
    float box_x = (pred_x - pad_w) / scale;
    float box_y = (pred_y - pad_h) / scale;

    int box_w = static_cast<int>(pred_w / scale);
    int box_h = static_cast<int>(pred_h / scale);

    int topLeftX = static_cast<int>(box_x - box_w / 2);
    int topLeftY = static_cast<int>(box_y - box_h / 2);

    topLeftX = max(0, min(topLeftX, resizedFrame.cols - 1));
    topLeftY = max(0, min(topLeftY, resizedFrame.rows - 1));
    box_w = min(box_w, resizedFrame.cols - topLeftX);
    box_h = min(box_h, resizedFrame.rows - topLeftY);

    cv::Rect roi(topLeftX, topLeftY, box_w, box_h);
 
    return roi;
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

void drawBoundingBox(cv::Mat img,cv::Rect roi){
    cv::rectangle(img,roi,cv::Scalar(255, 0, 0), 2);
}

void perClassNMS(const vector<cv::Rect>& boxes,const vector<float>& scores,const vector<int>& classIds,float scoreThreshold,float nmsThreshold,vector<int>& indices){
    // Group boxes by class
    map<int, vector<int>> classToIndices;
    for (size_t i = 0; i < classIds.size(); ++i) {
        if (scores[i] >= scoreThreshold) {
            classToIndices[classIds[i]].push_back(static_cast<int>(i));
        }
    }

    // Apply NMS for each class
    for (const auto& kv : classToIndices) {
        int cls = kv.first;
        const vector<int>& clsIndices = kv.second;

        // Collect class-specific boxes and scores
        vector<cv::Rect> clsBoxes;
        vector<float> clsScores;
        for (int idx : clsIndices) {
            clsBoxes.push_back(boxes[idx]);
            clsScores.push_back(scores[idx]);
        }

        // Perform OpenCV NMS
        vector<int> clsNmsIndices;
        cv::dnn::NMSBoxes(clsBoxes, clsScores, scoreThreshold, nmsThreshold, clsNmsIndices);

        // Map back to original indices
        for (int keptIdx : clsNmsIndices) {
            indices.push_back(clsIndices[keptIdx]);
        }
    }
}

void drawProcessedNMS(cv::Mat resizedFrame,vector<string> class_names,vector<int> indices,vector<cv::Rect> boundingBoxes,vector<float> confidences,vector<int> classIds, float scoreThreshold, float nmsThreshold){
    cv::dnn::NMSBoxes(boundingBoxes, confidences, scoreThreshold, nmsThreshold, indices);

    // Draw only the selected boxes
    for (int idx : indices) {
        cv::Rect box = boundingBoxes[idx];
        string classDetected = class_names[classIds[idx]];
        drawBoundingBox(resizedFrame, box);
        cv::putText(resizedFrame, classDetected, cv::Point(box.x, box.y - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
    }

}

int main() {

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
                            //cv::Scalar plateColor = plateDetection.color;

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
}

