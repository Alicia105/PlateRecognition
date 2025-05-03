#include <iostream>
#include <fstream>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
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

int main() {

    string pathToVideo="../videos/2103099-uhd_3840_2160_30fps.mp4";
    string pathToCarModel="../models/yolov8n.onnx";
    //string pathToPlateModel="../models/yolov8n.onnx";
    string classNameFilePath="../models/coco.names";
    vector<string> wantedClasses={"car","motorbike","bus","truck"};  

    //string pathToPlateModel="../models/yolov8n.onnx";

    cv::VideoCapture cap(pathToVideo);

    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video stream.\n";
        return -1;
    }

    cv::dnn::Net net = cv::dnn::readNetFromONNX(pathToCarModel);
    //cv::dnn::Net netPlate = cv::dnn::readNetFromONNX(pathToPlateModel);
    vector<string> class_names=getClassNames(classNameFilePath);

    float width=cap.get(cv::CAP_PROP_FRAME_WIDTH);
    float height=cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    float fps=cap.get(cv::CAP_PROP_FPS);

    int resized_width = 1280;
    int  resized_height = 720;

    float x_scale=static_cast<float>(resized_width) /width;
    float y_scale=static_cast<float>(resized_height) /height;
    
    cout<<"X_scale ="<<x_scale<<" , "<<"Y_scale ="<<y_scale<<endl;

    int treated=0;
    int dropped=0;
    int num=0;

    
    while (cap.isOpened()) {

        cv::Mat frame,resizedFrame;
        cap >> frame;
        cv::resize(frame,resizedFrame,cv::Size(resized_width,resized_height));

        if (frame.empty()) break;
        
        cv::Mat resized_rgb;
        cv::cvtColor(resizedFrame, resized_rgb, cv::COLOR_BGR2RGB);
        
        cv::Mat blob=cv::dnn::blobFromImage(resized_rgb, 1/255.0, cv::Size(640, 640), cv::Scalar(), true, false);
        //cv::Mat blobPlate = cv::dnn::blobFromImage(frame, 1/255.0, cv::Size(640, 640), cv::Scalar(), true, false);

        //Check models parameters
        cout << "Blob shape: [" << blob.size[0] << " x " << blob.size[1] << " x " << blob.size[2] << " x " << blob.size[3] << "]" << endl;
        cv::dnn::Layer* layer = net.getLayer(net.getLayerNames()[0]);  // Check the first layer
        cout << "Layer name: " << layer->name << endl;

        net.setInput(blob);
      
        vector<cv::Mat> outputs;
        vector<string> outLayerNames = net.getUnconnectedOutLayersNames();
        
        cout << "Output layer names: " << endl;
        for (const auto& layerName : outLayerNames) {
            cout << layerName << endl;
        }

        net.forward(outputs, outLayerNames);
        //net.forward(outputs, net.getUnconnectedOutLayersNames());

        cout << "Number of outputs: " << outputs.size() << endl;
        for (size_t i = 0; i < outputs.size(); i++) {
            cout << "Output " << i << " shape: [" 
                << outputs[i].rows << " x " 
                << outputs[i].cols << "]" << endl;
        }

        if (outputs.empty()) {
            cerr << "Error: Empty output from model." << endl;
            continue;
        }

        //cv::Mat output = net.forward();
        
        cv::Mat output = outputs[0];
        output = output.reshape(1, {84, 8400});  // 84 x 8400
        output = output.t();
        cout<<"Output : [rows="<<output.rows<<" x cols="<<output.cols<<"]"<<endl;

        float objectnessThreshold = 0.4;
        float scoreThreshold = 0.5;

        vector<cv::Rect> boundingBoxes;
        vector<float> confidences;
        vector<int> classIds;

        for(int i=0; i<output.rows; i++){
            float objectness_raw = output.at<float>(i, 4);
            float objectness = sigmoid(objectness_raw);
            
            cout<<"Objectness="<<objectness<<endl;

            if(objectness<objectnessThreshold){
                dropped++;
                continue;                
            }
            cv::Point classIdPoint;
            double confidence;

            cv::Mat classes_scores = output.row(i).colRange(5, output.cols).clone();
            for (int j = 0; j < classes_scores.cols; j++) {
                classes_scores.at<float>(0, j) = sigmoid(classes_scores.at<float>(0, j));
            }

            cv::minMaxLoc(classes_scores,0,&confidence,0,&classIdPoint);

            cout<<"Confidence="<<confidence<<endl;
          
            if (confidence > scoreThreshold) {

                string classDetected=class_names[classIdPoint.x];
                
                for(auto name : wantedClasses){
                    if (name==classDetected){
                        //Scaled bounding box
                        cv::Rect roi = getUnpaddedAndScaledBox(i,resizedFrame,output);
                        boundingBoxes.push_back(roi);
                        confidences.push_back(confidence);
                        classIds.push_back(classIdPoint.x);
                        /*cv::Mat blobPlate = cv::dnn::blobFromImage(resizedFrame, 1/255.0, cv::Size(640, 640), cv::Scalar(), true, false);
                        netPlate.setInput(blobPlate);
                        vector<cv::Mat> plateOutputs;
                        netPlate.forward(plateOutputs, netPlate.getUnconnectedOutLayersNames());
                        if (plateOutputs.empty()) {
                            std::cerr << "Error: Empty output from plate model." << std::endl;
                            continue;
                        }*/               
                        
                    }
                }              
                
                
                treated++;
            }
        }

        vector<int> indices;
        float nmsThreshold = 0.3;
        cv::dnn::NMSBoxes(boundingBoxes, confidences, scoreThreshold, nmsThreshold, indices);

        // Draw only the selected boxes
        for (int idx : indices) {
            cv::Rect box = boundingBoxes[idx];
            string classDetected = class_names[classIds[idx]];
            drawBoundingBox(resizedFrame, box);
            cv::putText(resizedFrame, classDetected, cv::Point(box.x, box.y - 10),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        }
        
        cv::imshow("Original", resizedFrame);
        num++;
        int k = cv::waitKey(10); // Wait for a keystroke in the window
        if(k=='q'){break;}
        if(num%5==0) continue;
    }

    cap.release();
    cv::destroyAllWindows();

    width=static_cast<int>(width);
    height=static_cast<int>(height);
    fps=static_cast<int>(fps);

    cout<<"Frame : [width="<<width<<" x height="<<height<<"]"<<endl;
    cout<<"FPS :"<<fps<<endl;
    cout<<"X_scale ="<<x_scale<<" , "<<"Y_scale ="<<y_scale<<endl;
    cout<<"Number of images ="<<num<<" , "<<"Treated ="<<treated<<" , "<<"Dropped ="<<dropped<<endl;

    return 0;
    
}

/*int main() {

    // Paths
    string pathToCarModel = "../models/yolov8n.onnx";
    string imagePath = "../images/test.jpg";
    string classNameFilePath="../models/coco.names";
    
    // Load network
    cv::dnn::Net net = cv::dnn::readNetFromONNX(pathToCarModel);
    vector<string> class_names=getClassNames(classNameFilePath);

    // Load image
   
    cv::Mat img = cv::imread(imagePath);
  
    if (img.empty()) {
        std::cerr << "Could not read the image: " << imagePath << std::endl;
        return 1;
    }

    int target_width = 640;
    int target_height = 640;

    float scale_x = target_width / (float)img.cols;
    float scale_y = target_height / (float)img.rows;

    float scale = min(scale_x, scale_y);
    int new_width = int(img.cols * scale);
    int new_height = int(img.rows * scale);

    cv::Mat img_resized;
    cv::resize(img, img_resized, cv::Size(new_width, new_height));

    cv::Mat img_padded(target_height, target_width, img_resized.type(), cv::Scalar(0, 0, 0));
    img_resized.copyTo(img_padded(cv::Rect(0, 0, new_width, new_height)));

    // Preprocess for model input
    cv::Mat blob = cv::dnn::blobFromImage(img_padded, 1.0 / 255.0, cv::Size(target_width,target_height), cv::Scalar(), true, false);

    // Set the input
    net.setInput(blob);

    // Forward pass
    vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());
    cv::Mat output=outputs[0];

    //cv::Mat output = net.forward();
    //cv::Mat output = outputs[0];
    cout<<"Output : [rows="<<output.rows<<" x cols="<<output.cols<<"]"<<endl;

    float objectnessThreshold = 0.1;
    float scoreThreshold = 0.4;
    
    for(int i=0; i<output.rows; i++){
        float objectness = output.at<float>(i,4);
        cout<<"Objectness="<<objectness<<endl;

        if(objectness<objectnessThreshold){
            continue;                
        }

        cv::Point classIdPoint;
        double confidence;

        cv::Mat classes_scores = output.row(i).colRange(5,output.cols);
        cv::minMaxLoc(classes_scores,0,&confidence,0,&classIdPoint);

        cout<<"Confidence="<<confidence<<endl;
        //vector<cv::Rect> boundingBoxes;

        if (confidence > scoreThreshold) {

            vector<int> coord = getRectCoordinates(i,output,img);
            
            //Scaled bounding box
            /*int scaled_topLeftX=static_cast<int>(round(coord[0]*img.cols));
            int scaled_topLeftY=static_cast<int>(round(coord[1]*img.rows));
            int scaled_width=static_cast<int>(round(coord[2]*img.cols));
            int scaled_height=static_cast<int>(round(coord[3]*img.rows));

            // Save box, class, and confidence
            
            cv::Rect roi(scaled_topLeftX,scaled_topLeftY,scaled_width,scaled_height);
            drawBoundingBox(img,roi);
            string label = class_names[classIdPoint.x] + " " + to_string(confidence).substr(0, 4);
            cv::putText(img, label, cv::Point(scaled_topLeftX, scaled_topLeftY - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);

            // Scale the bounding box back to original size
            int scaled_topLeftX = static_cast<int>(round(coord[0] / scale_x));
            int scaled_topLeftY = static_cast<int>(round(coord[1] / scale_y));
            int scaled_width = static_cast<int>(round(coord[2] / scale_x));
            int scaled_height = static_cast<int>(round(coord[3] / scale_y));

            // Account for padding by adjusting the coordinates
            int padded_topLeftX = scaled_topLeftX;
            int padded_topLeftY = scaled_topLeftY;

            // Draw the bounding box on the image
            cv::Rect roi(padded_topLeftX, padded_topLeftY, scaled_width, scaled_height);
            drawBoundingBox(img, roi);

            // Label the box with class name and confidence
            string label = class_names[classIdPoint.x] + " " + to_string(confidence).substr(0, 4);
            cv::putText(img, label, cv::Point(padded_topLeftX, padded_topLeftY - 5), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);

        }

    }
    
    cv::imshow("Original",img);
    cv::waitKey(0);

    return 0;
}*/


