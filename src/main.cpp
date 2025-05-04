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

    string pathToVideo="../videos/2103099-uhd_3840_2160_30fps.mp4";
    string pathToCarModel="../models/yolov8n.onnx";
    string pathToPlateModel="../models/yolov8n_plate.onnx";
    string classNameFilePath="../models/coco.names";
    vector<string> wantedClasses={"car"};  
    //vector<string> wantedClasses={"person","bicycle","car","motorbike","bus","truck"};  
    
    //string pathToPlateModel="../models/yolov8n.onnx";

    cv::VideoCapture cap(pathToVideo);

    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video stream.\n";
        return -1;
    }

    cv::dnn::Net net = cv::dnn::readNetFromONNX(pathToCarModel);
    cv::dnn::Net netPlate = cv::dnn::readNetFromONNX(pathToPlateModel);
    vector<string> class_names=getClassNames(classNameFilePath);
    vector<string> class_name_plate={"License_Plate"};

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

    //cv::Mat img = cv::imread(imagePath);

    
    while (cap.isOpened()) {

        cv::Mat frame,resizedFrame;
        cap >> frame;
        cv::resize(frame,resizedFrame,cv::Size(resized_width,resized_height));

        if (frame.empty()) break;
        
        cv::Mat resized_rgb;
        cv::cvtColor(resizedFrame, resized_rgb, cv::COLOR_BGR2RGB);
        
        cv::Mat blob=cv::dnn::blobFromImage(resizedFrame, 1/255.0, cv::Size(640, 640), cv::Scalar(), true, false);
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

        vector<cv::Rect> boundingBoxesPlates;
        vector<float> confidencesPlates;
        vector<int> classIdsPlates;

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
                        /*int pred_x = static_cast<int>(output.at<float>(i, 0));
                        int pred_y = static_cast<int>(output.at<float>(i, 1));
                        int pred_w = static_cast<int>(output.at<float>(i, 2));
                        int pred_h = static_cast<int>(output.at<float>(i, 3));

                        cv::Rect roi(pred_x,pred_y,pred_w,pred_h);*/
                        if(roi.width<=40 || roi.height<=40) continue;
                        confidence*=objectness;
                        boundingBoxes.push_back(roi);
                        confidences.push_back(confidence);
                        classIds.push_back(classIdPoint.x);
                        
                        cv::Mat imagePlate=resizedFrame(roi);
                        cv::Mat blobPlate = cv::dnn::blobFromImage(imagePlate, 1/255.0, cv::Size(640, 640), cv::Scalar(), true, false);
                        netPlate.setInput(blobPlate);
                        vector<cv::Mat> plateOutputs;
                        netPlate.forward(plateOutputs, netPlate.getUnconnectedOutLayersNames());
                        if (plateOutputs.empty()) {
                            cerr << "Error: Empty output from plate model." <<endl;
                            continue;
                        }
                        cv::Mat plateOutput = plateOutputs[0];
                        plateOutput = plateOutput.reshape(1, {5, 8400});  // 5 x 8400
                        plateOutput = plateOutput.t();
                        cout<<"Output plate: [rows="<<plateOutput.rows<<" x cols="<<plateOutput.cols<<"]"<<endl;

                        for(int i=0; i<plateOutput.rows; i++){
                            float plateConfidence_raw = output.at<float>(i, 4);
                            float plateConfidence = sigmoid(plateConfidence_raw);
                            cout<<"Plate Confidence="<<plateConfidence<<endl;
                            if(plateConfidence<objectnessThreshold){
                                continue;                
                            }
                            cv::Point classIdPointPlate;
                            classIdPointPlate.x=0;

                            cv::Rect roiPlate = getUnpaddedAndScaledBox(i,imagePlate,plateOutput);
                            cv::Rect corrected= cv::Rect(roiPlate.x+roi.x,roiPlate.y+roi.y,roiPlate.width,roiPlate.height);

                            boundingBoxesPlates.push_back(corrected);
                            confidencesPlates.push_back(plateConfidence);
                            classIdsPlates.push_back(classIdPointPlate.x);

                        }            
                        
                    }
                }     
                
                treated++;
            }
        }

        vector<int> indices;
        vector<int> indicesPlate;
        float nmsThreshold = 0.3;

        //perClassNMS(boundingBoxes,confidences,classIds,scoreThreshold,nmsThreshold,indices);
        drawProcessedNMS(resizedFrame,class_name_plate,indicesPlate,boundingBoxesPlates,confidencesPlates,classIdsPlates,scoreThreshold,nmsThreshold);
        /*for (int idx : indices) {
            cv::rectangle(resizedFrame, boundingBoxes[idx], cv::Scalar(255, 0, 0), 2);
            cv::putText(resizedFrame, class_names[classIds[idx]], boundingBoxes[idx].tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,255,0), 1);
        } */     

        cv::imshow("Original",resizedFrame);
        num++;
        int k = cv::waitKey(10); // Wait for a keystroke in the window
        if(k=='q'){break;}
        //if(num%5==0) continue;
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

    string pathToCarModel="../models/yolov8n.onnx";
    //string pathToPlateModel="../models/yolov8n_plate.onnx";
    string classNameFilePath="../models/coco.names";
    vector<string> wantedClasses={"person","bicycle","car","motorbike","bus","truck"};
    string imagePath = "../images/test.jpg";
    
    //string pathToPlateModel="../models/yolov8n.onnx";

    cv::Mat img = cv::imread(imagePath);
  
    if (img.empty()) {
        std::cerr << "Could not read the image: " << imagePath << std::endl;
        return 1;
    }

    cv::dnn::Net net = cv::dnn::readNetFromONNX(pathToCarModel);
    //cv::dnn::Net netPlate = cv::dnn::readNetFromONNX(pathToPlateModel);
    vector<string> class_names=getClassNames(classNameFilePath);
    //vector<string> wantedClasses=getClassNames(classNameFilePath);
    //vector<string> class_name_plate={"License_Plate"};

    int resized_width = 638;
    int  resized_height = 359;

    int treated=0;
    int dropped=0;
    int num=0;

    cv::Mat resizedImg;
        
    cv::resize(img,resizedImg,cv::Size(resized_width,resized_height));

    cv::Mat blob=cv::dnn::blobFromImage(resizedImg, 1/255.0, cv::Size(640, 640), cv::Scalar(), true, false);
        
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
        return 1;
    }
        
    cv::Mat output = outputs[0];
    output = output.reshape(1, {84, 8400});  // 84 x 8400
    output = output.t();
    cout<<"Output : [rows="<<output.rows<<" x cols="<<output.cols<<"]"<<endl;

    float objectnessThreshold = 0.5;
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
                    cv::Rect roi = getUnpaddedAndScaledBox(i,resizedImg,output);
                    /*int pred_x = static_cast<int>(output.at<float>(i, 0));
                    int pred_y = static_cast<int>(output.at<float>(i, 1));
                    int pred_w = static_cast<int>(output.at<float>(i, 2));
                    int pred_h = static_cast<int>(output.at<float>(i, 3));
                    cv::Rect roi(pred_x,pred_y,pred_w,pred_h);

                    confidence*=objectness;

                    if(roi.width<=40 || roi.height<=40) continue;
                    boundingBoxes.push_back(roi);
                    confidences.push_back(confidence);
                    classIds.push_back(classIdPoint.x);    
                        
                }     
                
                treated++;
            }
        }   
    }

    vector<int> indices;
    //vector<int> indicesPlate;
    float nmsThreshold = 0.2;
    float threshold=0.25;

    float nmsTh = 0.4;
    float th=0.3;

    /*for(auto r: boundingBoxes){
        drawBoundingBox(resizedImg,r);
    }

    vector<cv::Rect>clearedBoxes;
    vector<float> clearedConfidences;
    vector<int> clearedClassIds;
    perClassNMS(boundingBoxes,confidences,classIds,threshold,nmsThreshold,indices);

    for(int i : indices){
        clearedBoxes.push_back(boundingBoxes[i]);
        clearedConfidences.push_back(confidences[i]);
        clearedClassIds.push_back(classIds[i]);
    }
    
    vector<int> ind;

    drawProcessedNMS(resizedImg,class_names,ind,clearedBoxes,clearedConfidences,clearedClassIds,th,nmsTh);
    /*for (int idx : indices) {
        cv::rectangle(resizedImg, boundingBoxes[idx], cv::Scalar(255, 0, 0), 2);
        cv::putText(resizedImg, class_names[classIds[idx]], boundingBoxes[idx].tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,255,0), 1);
    }   

    cv::imshow("Original",resizedImg);
   
    int k = cv::waitKey(0) & 0xFF;  // mask to get the lower 8 bits (important on some systems)
    if (k == 'q') {
        cv::destroyAllWindows();
    }

    // Save the image to disk
    string s="../results/output.jpg";
    cv::imwrite(s, resizedImg);
    num++;
    cout<<"Number of images ="<<num<<" , "<<"Treated ="<<treated<<" , "<<"Dropped ="<<dropped<<endl;

    return 0;
}*/