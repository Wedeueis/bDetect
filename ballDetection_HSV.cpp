#include "opencv2/opencv.hpp"
#include <stdlib.h>
#include <iostream>
#include <string>
#include <json/json.h>
#include <fstream>
#include <vector>

//global variables
cv::Mat Gframe;
cv::Scalar mean_color(0,0,0);
double x_color = 0, y_color = 0;
int colorH = 10, colorS = 229, colorV = 255, rangeH = 10, rangeS = 30, rangeV = 81;
cv::Point2f fieldCorners[4];
double resize = 0.5;
int selectedCorner = 0;
char state = 'd';

void on_trackbar(int);
void changeCameraProp(std::string key, std::string value, Json::Value root);
void createTrackBars();
cv::Mat fieldCornersUpdated(cv::Point2f perspectiveIn[], cv::Size size);
void actionPickCorners(cv::VideoCapture &cap, Json::Value &root);
void actionConfigureColors(cv::VideoCapture &cap, Json::Value &root);
void CallBackFunc(int event, int x, int y, int flags, void* userdata);
void colorDetection(cv::Mat src, cv::Mat &mask, int it);
void findPos(cv::Mat &src,cv::Mat &tgt, std::vector<std::vector<cv::Point> > &contours,
             std::vector<cv::Vec4i> &hierarchy, Json::Value &root, float k);

int main(int, char**){
  //loading and testing json file
  Json::Value root;
	std::ifstream Config("configs.json");
	Config >> root;

	cv::Size fieldSize(root.get("fieldSize", 0)[0].asInt(),
                     root.get("fieldSize", 0)[1].asInt() );
  cv::Size fieldSizeMM(root.get("fieldSizeMM", 0)[0].asInt(),
                              root.get("fieldSizeMM", 0)[1].asInt() );
  int circleWarpSize = root.get("circleWarpSize", 0).asInt();

  // Constant that multiplied by a unit in mm get the size in pc
  float FIELD_MM = (float)(fieldSize.width) / (float)(fieldSizeMM.width);

  // Plotting (DEBUG)
  int plotWidth = int(fieldSize.width * 2);
  int plotHeigth = int(fieldSize.height * 2);
  std::vector<int> plots;
  //plotsLg = None;
  //plot = np.zeros((plotHeigth, plotWidth, 3), np.uint8)
  //cv2.resizeWindow("plots", plotWidth, plotHeigth)

  changeCameraProp("focus_auto", "0", root);
  changeCameraProp("exposure_auto_priority", "0", root);
  changeCameraProp("exposure_auto", "1", root);
  changeCameraProp("exposure_absolute", "75", root);
  createTrackBars();

  std::string camera = root.get("camera", 1).asString();
  char capNumber = camera.at(10) - 48;
  cv::VideoCapture cap(capNumber); // open the default camera
  if (!cap.isOpened())  // check if we succeeded
    return -1;

  cap.set(3, root.get("cameraRes",0)[0].asInt());
  cap.set(4, root.get("cameraRes",0)[1].asInt());
  cv::Size frameSize(cap.get(3), cap.get(4) );

  //
  // Initialize Static Masks
  // This mask is used to exclude outer area of circles detected in HoughCircles
  //
  cv::Mat circleMask(circleWarpSize,circleWarpSize,CV_8UC1,cv::Scalar(1,1,1));
  int halfMaskSize = circleWarpSize / 2;
  cv::circle(circleMask,
            //center
            cv::Point(circleWarpSize / 2, circleWarpSize / 2),
            //radius
            (int)(circleWarpSize / 2.5),
            cv::Scalar(255,255,255), -1, 8 , 0 );

  cv::Scalar color_rgb(0,0,0);
  fieldCorners[0] = cv::Point2f(root.get("fieldCorners",0)[0][0].asFloat(),
                                      root.get("fieldCorners",0)[0][1].asFloat() );
  fieldCorners[1] = cv::Point2f(root.get("fieldCorners",0)[1][0].asFloat(),
                                      root.get("fieldCorners",0)[1][1].asFloat() );
  fieldCorners[2] = cv::Point2f(root.get("fieldCorners",0)[2][0].asFloat(),
                                      root.get("fieldCorners",0)[2][1].asFloat() );
  fieldCorners[3] = cv::Point2f(root.get("fieldCorners",0)[3][0].asFloat(),
                                      root.get("fieldCorners",0)[3][1].asFloat() );
  /*
  cv::Scalar colors[6] = {cv::Scalar( root.get("colors",0)["blue"][0].asInt() ,
                    root.get("colors",0)["blue"][1].asInt() ,
                    root.get("colors",0)["blue"][2].asInt() ),
       cv::Scalar( root.get("colors",0)["green"][0].asInt() ,
                    root.get("colors",0)["green"][1].asInt() ,
                    root.get("colors",0)["green"][2].asInt() ),
       cv::Scalar( root.get("colors",0)["orange"][0].asInt() ,
                    root.get("colors",0)["orange"][1].asInt() ,
                    root.get("colors",0)["orange"][2].asInt() ),
       cv::Scalar( root.get("colors",0)["purple"][0].asInt() ,
                    root.get("colors",0)["purple"][1].asInt() ,
                    root.get("colors",0)["purple"][2].asInt() ),
       cv::Scalar( root.get("colors",0)["red"][0].asInt() ,
                    root.get("colors",0)["red"][1].asInt() ,
                    root.get("colors",0)["red"][2].asInt() ),
       cv::Scalar( root.get("colors",0)["yellow"][0].asInt() ,
                    root.get("colors",0)["yellow"][1].asInt() ,
                    root.get("colors",0)["yellow"][2].asInt()),
                  };
  */
  cv::Mat warpMatrix =  fieldCornersUpdated(fieldCorners, frameSize);

  for(;;){
    /*
    if(k == ord('s')):
      if(SHOW_DISPLAY is True):
        SHOW_DISPLAY = False;
      else:
        SHOW_DISPLAY = True;

    #
    # Check for "calibrate ball" action
    #
    if (k == ord("b")):
      actionCalibrateBall();

    #
    # Check for calibrate color action
    #
    if(k == ord('c')):
      actionConfigureColors();

    t = Timing();
    plots = [];
    plotsLg = None;

  #############################################################################################################

  #############################################################################################################


    # List where all objects get stored
    objects = [];

    # Capture frame
    t.start("capture")
    ret, frame = cap.read()
    height, width = frame.shape[:2];
    # plots.append(frame);
    out = frame;
    t.end();

    #
    # Generate Perspective Transform
    #
    t.start("wrap");
    field = cv2.warpPerspective(out, warpMatrix, tuple(fieldSize));
    plots.append(field);
    out = field;
    t.end();
    */
    cv::Mat bin;
    cap >> Gframe; // get a new frame from camera
    cv::Mat warpedFrame = Gframe.clone();
    cv::warpPerspective(Gframe,warpedFrame,warpMatrix,Gframe.size(),cv::INTER_NEAREST,
                        cv::BORDER_CONSTANT, cv::Scalar() );
    Gframe = warpedFrame;
    warpedFrame.release();

    colorDetection(Gframe, bin, 2);

    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;

    findPos(bin, Gframe, contours, hierarchy, root, FIELD_MM);

    //show results
    //Create a window
    cv::namedWindow("Frame", CV_WINDOW_AUTOSIZE);

    //set the callback function for any mouse event
    cv::setMouseCallback("Frame", CallBackFunc, NULL);

    cv::imshow("Frame",Gframe);

    int k = cv::waitKey(30);
    if (k == 27 || k == 'q')
      break;
    else if(k == 'f'){
      state = 'f';
      actionPickCorners(cap, root);
      warpMatrix = fieldCornersUpdated(fieldCorners, frameSize);
      cv::warpPerspective(Gframe,warpedFrame,warpMatrix,fieldSize,cv::INTER_LINEAR,
                          cv::BORDER_CONSTANT, cv::Scalar() );
    }else if(k == 'c'){
      state = 'c';
      actionConfigureColors(cap,root);
    }else if(k == 's') {
      //função de escrever no json
      std::ofstream configs;
      configs.open("configs.json");

      Json::StyledWriter styledWriter;
      configs << styledWriter.write(root);

      configs.close();
    }
  }
  return 0;
}

void changeCameraProp(std::string key, std::string value, Json::Value root){
	if (system(NULL)) puts ("Ok");
  else exit (EXIT_FAILURE);
  std::string camera = root.get("camera", 1).asString();
  std::string s ="v4l2-ctl -d " + camera + "-c " + key + "=" + value;
  system(s.c_str());
}

void createTrackBars(){
	//create the trackbars
	cv::namedWindow("TrackBar", 1);
	cvCreateTrackbar("HUE", "TrackBar", &colorH, 255, on_trackbar);
	cvCreateTrackbar("SATURATION", "TrackBar", &colorS, 255, on_trackbar);
	cvCreateTrackbar("VALUE", "TrackBar", &colorV, 255, on_trackbar);
	cvCreateTrackbar("RANGE_H", "TrackBar", &rangeH, 128, on_trackbar);
	cvCreateTrackbar("RANGE_S", "TrackBar", &rangeS, 128, on_trackbar);
	cvCreateTrackbar("RANGE_V", "TrackBar", &rangeV, 128, on_trackbar);
}

void on_trackbar(int){};

//Function to create a color mask and "cut" the ball in the source image
void colorDetection(cv::Mat src, cv::Mat &mask, int it){
	cv::Mat hsv, tgt, thrs;
	//3-channel binary mask
	cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);
	cv::GaussianBlur(hsv, hsv, cv::Size(5, 5),0,0);
	cv::inRange(hsv, cv::Scalar(colorH - rangeH, colorS - rangeS, colorV - rangeV), cv::Scalar(colorH + rangeH + 1 , colorS + rangeS +
	1, colorV + rangeV + 1), mask);

	//image erosion
	cv::Mat element = cv::getStructuringElement( 0,cv::Size( 2,2 ),cv::Point( 0, 0 ) );
  cv::erode( mask, mask, element, cv::Point( 0, 0 ), it	);
	cv::dilate( mask, mask, element, cv::Point( 0, 0 ), 2*it +1);

	//mask aplication
	cv::Mat mask3[] = { mask,mask,mask };
	cv::merge(mask3, 3, thrs);
	cv::bitwise_and(thrs, src, tgt);

	cv::imshow("Bola", tgt);
	//cv::imshow("HSV", hsv);

}

//Function to find the ball position in the screen
void findPos(cv::Mat &src,cv::Mat &tgt, std::vector<std::vector<cv::Point> > &contours,
              std::vector<cv::Vec4i> &hierarchy, Json::Value &root, float k){
	cv::Mat temp = src.clone();

	if( !temp.empty())
		cv::findContours(temp,contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
		//select best contour
		std::vector<std::vector<cv::Point> > contours_poly( contours.size() );
		std::vector<cv::Point2f> center( contours.size() );
		std::vector<float> radius( contours.size() );

		int realBallRadius = root.get("ball_radius", 0).asInt();
    int bestBallRadiusDif = 0, final_radius, bestBall = 0;
		cv::Point ball_center;
		std::vector<cv::Point> hull;
		double per, cntArea, relation, radiusDif;

		for( int i = 0; i < contours.size(); i++ ){
			//cv::approxPolyDP( cv::Mat(contours[i]), contours_poly[i], 3, true );//verificar se eh necessario
			cv::minEnclosingCircle( (cv::Mat)contours[i], center[i], radius[i] );
			// per = arcLength(contours[i], true);
			// cv::convexHull(contours[i], hull, true);
			// cntArea = cv::contourArea(hull,false) + 0.1;
			// relation = (per*radius[i])/2*cntArea;
      radius[i] /= k;
			radiusDif = abs(realBallRadius - radius[i]);
			if(bestBall == 0 || radiusDif < bestBallRadiusDif){
				final_radius = (int)radius[i];
				ball_center.x = (int)center[i].x;
				ball_center.y = (int)center[i].y;
				bestBall = i;
				bestBallRadiusDif = radiusDif;
			}
		}

		if(bestBall != 0){
			cv::circle( tgt, center[bestBall], (int)radius[bestBall], cv::Scalar(255,0,0), 2, 8, 0 );
      root["ball_radius"] = final_radius;
      root["ball_x"] = (int)(center[bestBall].x/k);
      root["ball_y"] = (int)(center[bestBall].y/k);
      std::ofstream configs;
      configs.open("configs.json");
      Json::StyledWriter styledWriter;
      configs << styledWriter.write(root);
      configs.close();
		}
}

//function to treat the events (mouse click) in the created windows
void CallBackFunc(int event, int x, int y, int flags, void* userdata) {
    switch (state) {
      case 'c':
    		if (event == cv::EVENT_LBUTTONUP) {
    			//Get click x y position to a global variable
    			x_color = x;
    			y_color = y;
          int offset = 10;
    			int x1 = int(x - offset), x2 = int(x + offset);
    			int y1 = int(y - offset), y2 = int(y + offset);

    			//Check if it exeeds the boundaries
    			if(x1 < 0 or y1 < 0 or  x2 >= Gframe.cols or y2 >= Gframe.rows) {
           	std::cout << "!!! Exeecds boundaries" << std::endl;
    				return;
          }

    			//Get Region of Interest
    			cv::Rect roi(x1,y1,x2-x1, y2-y1);
          cv::Mat image_roi = Gframe(roi);
    			cv::cvtColor(image_roi,image_roi, cv::COLOR_BGR2Lab, 0);

          //create a round mask
          cv::Mat mask_pickcolor(offset*2,offset*2,CV_8UC1,cv::Scalar(1,1,1));
          cv::circle(mask_pickcolor,
                    //center
                    cv::Point(offset, offset),
                    //radius
                    offset,
                    cv::Scalar(255,255,255), -1, 8 , 0 );

    			//Blur Image
          cv::GaussianBlur(image_roi, image_roi, cv::Size(3, 3),0,0);

    			//Find Mean of colors (Excluding outer areas)
    			mean_color = cv::mean(image_roi,mask_pickcolor);
        }
        break;
      case 'f':
         if  ( event == cv::EVENT_LBUTTONDOWN ) {
           fieldCorners[selectedCorner].x = (x / resize);
           fieldCorners[selectedCorner].y = (y / resize);
         }
        break;
    }
}

//function to create the matrix to change the perspective of the image
cv::Mat fieldCornersUpdated(cv::Point2f perspectiveIn[], cv::Size fieldSize){
  cv::Point2f perspectiveOut[] = { cv::Point2f(0.0, 0.0),
                                 cv::Point2f(fieldSize.width , 0.0),
                                 cv::Point2f(fieldSize.width, fieldSize.height),
                                 cv::Point2f( 0.0, fieldSize.height) };
  for(int i = 0; i < 4; i++){
    std::cout << perspectiveIn[i].x << " " << perspectiveIn[i].y << std::endl;
  }

  for(int i = 0; i < 4; i++){
    std::cout << perspectiveOut[i].x << " " << perspectiveOut[i].y << std::endl;
  }

  return cv::getPerspectiveTransform(perspectiveIn, perspectiveOut);
}

//function to choose the appropriate corner points to the field
void actionPickCorners(cv::VideoCapture &cap, Json::Value &root) {
  cv::namedWindow("pickCorners");
  cv::setMouseCallback("pickCorners", CallBackFunc);
  for(;;) {
        int k = cv::waitKey(30);
        if (k == 27 || k == 'q'){
          state = 'd';
          break;
        }else if(k == 's') {
          std::ofstream configs;
          configs.open("configs.json");
          Json::StyledWriter styledWriter;
          configs << styledWriter.write(root);
          configs.close();
        }else if(k >= 49 and k <= 52)  //Change the selected corner on press 1-4 key
          selectedCorner = k - 49;
        else if(k >= 73 and k <= 76) { //Move 1px when
          switch(k){
            case 73:
              fieldCorners[selectedCorner].y += -1;
              std::cout << fieldCorners[selectedCorner].y << std::endl;
              break;
            case 74:
              fieldCorners[selectedCorner].x += -1;
              break;
            case 75:
              fieldCorners[selectedCorner].y += 1;
              break;
            case 76:
              fieldCorners[selectedCorner].x += 1;
              break;
          }
        }

        cv::Mat frame;
        cap >> frame;

        //Resize Frame
        cv::resize(frame, frame, cv::Size((int)(frame.cols*resize),(int)(frame.rows*resize) ),
                                                 0, 0, cv::INTER_AREA);
        cv::resizeWindow("pickCorners", frame.cols, frame.rows);

        // Change status text
        std::stringstream ss;
        ss << (selectedCorner + 1);
        std::string status( "Pick corner " + ss.str() );
        cv::putText(frame, status, cv::Point(10,30), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                    cv::Scalar(255, 0, 0), 2, cv::LINE_8, false);

        //Draw outline and highlight the selected corner
        for(int i = 0; i < 4; i++) {
          cv::Point2f cornerFrom( (int)(fieldCorners[i].x*resize),
                              (int)(fieldCorners[i].y*resize) );

          cv::Point2f cornerTo( (int)(fieldCorners[(i + 1) % 4].x*resize),
                            (int)(fieldCorners[(i + 1) % 4].y*resize) );

          cv::line(frame,
            cornerFrom,
            cornerTo,
            cv::Scalar(0, 0, 255), 1,
            cv::LINE_8, 0 );

          if (i == selectedCorner) {
            cv::circle(
              frame,
              cornerFrom,
              3, cv::Scalar(0, 255, 0), 1,
              cv::LINE_8, 0 );
          }

          cv::imshow("pickCorners", frame);
        }
  }

    cv::destroyWindow("pickCorners");

    for(int i = 0; i<4; i++){
      root["fieldCorners"][i][0] = fieldCorners[i].x;
      root["fieldCorners"][i][1] = fieldCorners[i].y;
    }

}

void actionConfigureColors(cv::VideoCapture &cap, Json::Value &root) {

  int selectedColor = 0;
  std::string Scolor = "blue";
  //Radius of measure Color mean
  int offset = 10;

  cv::namedWindow("pickColors");
  cv::setMouseCallback("pickColors", CallBackFunc);

  for(;;) {
    cv::Mat frame;
    cap >> frame;

    switch(selectedColor) {
      case 0: {
        Scolor = "blue";
        std::string status( "Pick color blue");
        cv::putText(frame, status, cv::Point(10,30), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                    cv::Scalar(255, 0, 0), 2, cv::LINE_8, false);
        break;
      }
      case 1: {
        Scolor = "green";
        std::string status( "Pick color green");
        cv::putText(frame, status, cv::Point(10,30), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                    cv::Scalar(255, 0, 0), 2, cv::LINE_8, false);
        break;
      }
      case 2:{
        Scolor = "orange";
        std::string status( "Pick color orange");
        cv::putText(frame, status, cv::Point(10,30), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                    cv::Scalar(255, 0, 0), 2, cv::LINE_8, false);
        break;
      }
      case 3: {
        Scolor = "purple";
        std::string status( "Pick color purple");
        cv::putText(frame, status, cv::Point(10,30), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                    cv::Scalar(255, 0, 0), 2, cv::LINE_8, false);
        break;
      }
      case 4: {
        Scolor = "red";
        std::string status( "Pick color red");
        cv::putText(frame, status, cv::Point(10,30), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                    cv::Scalar(255, 0, 0), 2, cv::LINE_8, false);
        break;
      }
      case 5:{
        Scolor = "yellow";
        std::string status( "Pick color yellow");
        cv::putText(frame, status, cv::Point(10,30), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                    cv::Scalar(255, 0, 0), 2, cv::LINE_8, false);
        break;
      }
    }

    //Get LAB Color of Mouse point
    std::stringstream ss1, ss2, ss3;
    int color_lab_L = round(mean_color[0]);
    ss1 << color_lab_L;
    int color_lab_A = round(mean_color[1]);
    ss2 << color_lab_A;
    int color_lab_B = round(mean_color[2]);
    ss3 << color_lab_B;
    std::string color_lab("["+ ss1.str() +","+ ss2.str() +","+ ss3.str() +"]");

    cv::putText(frame, color_lab, cv::Point(10,60), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                cv::Scalar(55, 255, 55), 2, cv::LINE_8, false);

    cv::circle(frame,cv::Point(x_color,y_color),offset,cv::Scalar(255,255,0),1);

    cv::imshow("pickColors", frame);

    int k = cv::waitKey(30);
    if( k==27 || k == 'q') {
      state = 'd';
      break;
    }else if( k >= 49 && k <= 53) {
      selectedColor = k - 49;
    }else if(k == 's') {
      //função de gravar no json
      root["colors"][Scolor][0] = color_lab_L;
      root["colors"][Scolor][1] = color_lab_A;
      root["colors"][Scolor][2] = color_lab_B;
      std::ofstream configs;
      configs.open("configs.json");
      Json::StyledWriter styledWriter;
      configs << styledWriter.write(root);
      configs.close();
    }
  }
  cv::destroyWindow("pickColors");

}
