#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include <map>
#include <bits/stdc++.h>
#include <opencv2/opencv.hpp>
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/core.hpp>
#include <opencv2/video/tracking.hpp>
#include "../include/database.hpp"
#include "../external/inference/inference.h"
#include "../external/sort/include/sort.h"

using namespace std;
using namespace sort;

const string alphabet = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ_";

void updateSavedPlates(map<string, int>& plates, string license) {
    if (license.length() != 7) return;

    if (plates.count(license)) {
        plates.at(license)++;
    } else {
        plates[license] = 1;
    }
}

void updateSavedVehicles(map<string,string>& vehicles, string license, string vehicleType){
    if(!vehicles.count(license) && license.length()==7){
        vehicles[license]= vehicleType;
    }
}

void cleanSavedPlates(map<string, int>& plates) {
    for (auto it = plates.begin(); it != plates.end(); ) {
        if (it->second <= 5) {
            it = plates.erase(it);
        } else {
            ++it;
        }
    }
}

void cleanSavedVehicles(const map<string, int>& plates, map<string, string>& vehicles) {
    for (auto it = vehicles.begin(); it != vehicles.end(); ) {
        if (!plates.count(it->first)) {
            it = vehicles.erase(it);
        } else {
            ++it;
        }
    }
}

bool savePlateInDatabase(sqlite3* db, const map<string, int>& plates, const map<string, string>& vehicles) {
    bool allSucceeded = true;

    for (auto plate : plates) {
        if (plate.second > 5) {
            auto it = vehicles.find(plate.first);
            if (it != vehicles.end()) {
                bool r = insertPlate(db, it->second, plate.first);
                allSucceeded = allSucceeded && r;
            }
        }
    }

    return allSucceeded;
}

string generateName(){
    time_t timestamp = time(nullptr);
    struct tm datetime;

    if (localtime_s(&datetime, &timestamp) != 0) {
        cerr << "ERROR: localtime_s() failed"<<endl;
    }

    char output[50];
    size_t count = strftime(output, sizeof(output), "%d-%m-%Y_%H-%M-%S", &datetime);
    if (count == 0) {
        cerr << "ERROR: strftime() failed"<<endl;
    }

    cout << "Formatted time: " << output << endl;

    string name = "../results/output_" + string(output)+ ".mp4";
    return name;
}

// Preprocess the plate image to match model input
cv::Mat preprocessPlate(const cv::Mat& plate) {
    cv::Mat gray, resized;
    cv::cvtColor(plate, gray, cv::COLOR_BGR2GRAY);
    cv::resize(gray, resized, cv::Size(140, 70));
    cv::Mat blob = cv::dnn::blobFromImage(resized, 1.0, cv::Size(140, 70), cv::Scalar(), false, false, CV_8U); // NCHW
    blob=blob.reshape(1, {1, 70, 140, 1});
    //cout << "Blob type: " << blob.type() <<endl;
    return blob;
}

// Decode the model output
string decodeOutput(const cv::Mat& flat_output, const string& alphabet, float& avg_confidence) {
    const int num_slots = 9;
    const int num_classes = 37;
    std::string result;
    std::vector<float> confidences;

    // Reshape the flat [1, 333] to [9, 37]
    cv::Mat reshaped = flat_output.reshape(1, num_slots);  // shape becomes [9, 37

    for (int i = 0; i < num_slots; ++i) {
        cv::Mat scores = reshaped.row(i);
        cv::Point classIdPoint;
        double maxVal;
        cv::minMaxLoc(scores, 0, &maxVal, 0, &classIdPoint);
        int classId = classIdPoint.x;

        char c = alphabet[classId];
        if (c != '_') {
            result += c;
            confidences.push_back(static_cast<float>(maxVal));  // Softmax value
        }
    }

    // Calculate average confidence
    avg_confidence = confidences.empty() ? 0.0f : accumulate(confidences.begin(), confidences.end(), 0.0f) / confidences.size();

    return result;
}

pair<string, float> recognizePlate(cv::dnn::Net& net, const cv::Mat& plate, const string& alphabet) {
    cv::Mat inputBlob = preprocessPlate(plate);
    //cout << "Blob shape: " << inputBlob.size << " Channels: " << inputBlob.channels() << endl;
    net.setInput(inputBlob);
    cv::Mat output = net.forward(); // shape [1, 9, 37], flattened to [9, 37]
   
    // Print number of dimensions
    //cout << "Output dims: " << output.dims << endl;

    // Print each dimension
    /*cout << "Output shape: [";
    for (int i = 0; i < output.dims; ++i) {
        std::cout << output.size[i];
        if (i < output.dims - 1) cout << ", ";
    }
    std::cout << "]" << std::endl;*/

    float confidence = 0.0f;
    string plateText = decodeOutput(output, alphabet, confidence);
    return {plateText, confidence};
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

    const string dbName = "../data/plates.db";

    if (createDatabase(dbName)) {
        cout << "Database created successfully." << endl;
    }
    else return 1;

    sqlite3* db;
    if (!openDatabase(dbName, &db)) return 1;

    bool runOnGPU = false;
    bool saveVideo = true;
    string pathToVideo="../videos/2103099-uhd_3840_2160_30fps.mp4";
    string pathToCarModel="../models/yolov8n.onnx";
    string pathToPlateModel="../models/yolov8n_plate.onnx";
    string pathToOcr="../models/global_mobile_vit_v2_ocr.onnx";
    string classNameFilePath="../models/coco.names";
    string classPlateFilePath="../models/plate.names"; 

    vector<string> allClasses=getClassNames(classNameFilePath);
    vector<string> wantedClasses={"person","bicycle","car","motorbike","bus","truck"}; 
    vector<cv::Scalar> savedColors;

    map<string,int> savedPlates;
    map<string,string> savedVehicles;
    map<int, pair<string, float>> bestPlates; //map<int, pair<string, float>> :tracker_id :int, plate:string, confidence:float


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

    string nameFile=generateName();

    cv::VideoWriter writer(nameFile, cv::VideoWriter::fourcc('m','p','4','v'),fps,cv::Size(resized_width,resized_height));

    if (!writer.isOpened()) {
        cerr << "Error: Cannot open the output file to write" << endl;
        saveVideo=false;
    }

    int numColors=10; //number of colors to draw boxes
    int numberOfFrame=0;

    float plateConfidenceThreshold=0.5;
    float confidenceThreshold=0.7;

    Sort::Ptr sortTracker = make_shared<Sort>(1, 3, 0.3f);

    while (cap.isOpened()){

        cv::Mat frame,resizedFrame,detectionsMat;;
        cap >> frame;

        if (frame.empty()) break;

        cv::resize(frame,resizedFrame,cv::Size(resized_width,resized_height));
        numberOfFrame++;
      
        //if(numberOfFrame%5==0)  continue;

        if(numberOfFrame%25==0){ 
            cleanSavedPlates(savedPlates);
            cleanSavedVehicles(savedPlates,savedVehicles);
            if(!savePlateInDatabase(db,savedPlates,savedVehicles)){
                cerr << "Plates not inserted successfully"<<endl;
            }
        }
        
        vector<Detection> output = inf.runInference(resizedFrame);
        int detections = output.size();
        std::cout << "Number of detections:" << detections <<endl;


        //Detection
        //input of the tracker update: detections: [xc, yc, w, h, score, class_id]
        for(int i=0; i<detections; ++i){
            Detection detection = output[i];

            for(auto name : wantedClasses){
                if (name==detection.className && detection.confidence>confidenceThreshold){

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

            // Ensure ROI is within image bounds
            int plates;
            vector<Detection> outputPlate;
            bbox &= cv::Rect(0, 0, resizedFrame.cols, resizedFrame.rows);

            if (bbox.width > 0 && bbox.height > 0) {
                //Plate detection
                cv::Mat imagePlate=resizedFrame(bbox);//safe crop
                outputPlate = infPlate.runInference(imagePlate);
                plates=outputPlate.size();
            }
            else{
                continue;
            }


            for(int k=0; k<plates; ++k){
                Detection plateDetection=outputPlate[k];

                if(plateDetection.confidence<plateConfidenceThreshold) continue;

                cv::Rect plateBox = plateDetection.box;
                cv::Scalar plateColor = color;
                            
                plateBox.x+=bbox.x;
                plateBox.y+=bbox.y;

                //Plate Detection box
                cv::rectangle(resizedFrame, plateBox, plateColor, 2);                        
                
                if(plateDetection.confidence>0.7){
                    //OCR in Plate Detection box 
                    cv::Mat plateCropped = resizedFrame(plateBox);
                    auto [text, conf] = recognizePlate(ocrNet,plateCropped,alphabet);
                    float highestConfidence = 0.5;
                    if(!bestPlates.count(tracker_id) && text.length() == 7 && conf>highestConfidence){
                        bestPlates[tracker_id] = {text, conf};
                    }
                    
                    //map<int, pair<string, float>> :tracker_id :int, plate:string, confidence:float
                    if (text.length() == 7 && conf > bestPlates[tracker_id].second) {
                        bestPlates[tracker_id] = {text, conf};
                    }

                    if (bestPlates.count(tracker_id)) {
                        string plateText=bestPlates[tracker_id].first; 
                        cv::Size textSizePlate = cv::getTextSize(plateText, cv::FONT_HERSHEY_DUPLEX, 1, 2, 0);
                        cv::Rect textBoxPlate(plateBox.x, plateBox.y - 40, textSizePlate.width + 10, textSizePlate.height + 20);

                        //Draw plate detection + OCR
                        cv::rectangle(resizedFrame, textBoxPlate, plateColor, cv::FILLED);
                        cv::putText(resizedFrame, plateText, cv::Point(plateBox.x + 5, plateBox.y - 10), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0, 0), 2, 0);
                        updateSavedPlates(savedPlates,plateText);
                        updateSavedVehicles(savedVehicles,plateText,className);      
                    }

                    //cout<<text<<endl;
                }
            }  
        }

        cout<<"Number of frames ="<<numberOfFrame<<endl;
        
        if(saveVideo) writer.write(resizedFrame);

        cv::imshow("Plate Recognition",resizedFrame);
        int k = cv::waitKey(10); // Wait for a keystroke in the window
        if(k=='q'){break;}
    }

    cap.release();
    writer.release();
    cv::destroyAllWindows();

    width=static_cast<int>(width);
    height=static_cast<int>(height);
    fps=static_cast<int>(fps);

    cout<<"Frame : [width="<<width<<" x height="<<height<<"]"<<endl;
    cout<<"FPS :"<<fps<<endl;
    cout<<"Number of video frames processed="<<numberOfFrame<<endl;

    printPlates(db);
    sqlite3_close(db);

    return 0;
}

/*void testCodec(const string& codecName, int fourcc, const string& filename) {
    cv::VideoWriter writer;
    writer.open(filename, fourcc, 30.0, cv::Size(640, 480));
    if (writer.isOpened()) {
        cout << codecName << " supported " << endl;
        writer.release();
        remove(filename.c_str()); // cleanup
    } else {
        cout << codecName << " not supported " << endl;
    }
}

int main() {
    testCodec("MP4V", cv::VideoWriter::fourcc('M','P','4','V'), "test.mp4");
    testCodec("XVID", cv::VideoWriter::fourcc('X','V','I','D'), "test.avi");
    testCodec("MJPG", cv::VideoWriter::fourcc('M','J','P','G'), "test.avi");
    testCodec("AVC1", cv::VideoWriter::fourcc('A','V','C','1'), "test.mp4");
    testCodec("H264", cv::VideoWriter::fourcc('H','2','6','4'), "test.mp4");
    return 0;
}*/
