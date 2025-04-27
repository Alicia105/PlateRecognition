#include <iostream>
#include <fstream>
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
    string classNameFilePath="../models/coco.names";
    //string pathToPlateModel="../models/yolov8n.onnx";

    cv::VideoCapture cap(pathToVideo);

    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video stream.\n";
        return -1;
    }

    cv::dnn::Net net = cv::dnn::readNetFromONNX(pathToCarModel);
    int width=0;
    int height=0;
    int fps=0;
    vector<string> class_names=getClassNames(classNameFilePath);
   
    while (cap.isOpened()) {
        
        width=static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
        height=static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
        fps=static_cast<int>(cap.get(cv::CAP_PROP_FPS));

        cv::Mat frame;
        cap >> frame;

        //if (frame.empty()) break;
        
        cv::Mat blob = cv::dnn::blobFromImage(frame, 1/255.0, cv::Size(640, 640), cv::Scalar(), true, false);
        net.setInput(blob);
      
        vector<cv::Mat> outputs;
        net.forward(outputs, net.getUnconnectedOutLayersNames());

        cv::Mat output = outputs[0];
        float objectnessThreshold = 0.7;

        for(int i=0; i<output.rows; i++){
            float objectness = output.at<float>(i,4);
            float scoreThreshold = 0.7;

            if(objectness<objectnessThreshold){
                continue;                
            }
            cv::Point classIdPoint;
            double confidence;

            cv::Mat classes_scores = output.row(i).colRange(5,85);
            minMaxLoc(classes_scores,0,&confidence,0,&classIdPoint);
            //vector<cv::Rect> boundingBoxes;

            if (confidence > scoreThreshold) {
                int centerX = (int)(output.at<float>(i, 0) * frame.cols);
                int centerY = (int)(output.at<float>(i, 1) * frame.rows);
                int width   = (int)(output.at<float>(i, 2) * frame.cols);
                int height  = (int)(output.at<float>(i, 3) * frame.rows);
        
                int topLeftX = centerX - width / 2;
                int topLeftY = centerY - height / 2;
        
                // Save box, class, and confidence

                string classDetected=class_names[classIdPoint.x];
                cv::Rect roi(topLeftX,topLeftY,width,height);
                //boundingBoxes.push_back(roi);
                drawBoundingBox(frame,roi);
            }
        }
        
        cv::imshow("Original", frame);

        int k = cv::waitKey(1); // Wait for a keystroke in the window
        if(k=='q'){break;}
    }

    cap.release();
    cv::destroyAllWindows();
    cout<<"Frame : [width="<<width<<" x height="<<height<<"]"<<endl;
    cout<<"FPS :"<<fps<<endl;
    return 0;
    
}

