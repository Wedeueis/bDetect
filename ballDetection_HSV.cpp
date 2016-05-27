#include "opencv2/opencv.hpp"
#include <stdlib.h>
#include <iostream>
#include <string>
#include <json/json.h>
#include <fstream>
#include <vector>

//global variables for pick color function
cv::Scalar myints(16,2,77,29);
int x_color = 0, y_color = 0;
int colorH = 10, colorS = 229, colorV = 255, rangeH = 10, rangeS = 30, rangeV = 81;
cv::Point2f fieldCorners[4];
double resize = 0.5;
int selectedCorner = 0;
char state = 'c';

void on_trackbar(int);
void changeCameraProp(std::string key, std::string value);
void createTrackBars();
cv::Mat fieldCornersUpdated(cv::Point2f perspectiveIn[], cv::Size size);
void actionPickCorners(cv::VideoCapture &cap);
void CallBackFunc(int event, int x, int y, int flags, void* userdata);
void colorDetection(cv::Mat src, cv::Mat &mask, int it);
void findPos(cv::Mat &src,cv::Mat &tgt, std::vector<std::vector<cv::Point> > &contours, std::vector<cv::Vec4i> &hierarchy);

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

  changeCameraProp("focus_auto", "0");
  changeCameraProp("exposure_auto_priority", "0");
  changeCameraProp("exposure_auto", "1");
  changeCameraProp("exposure_absolute", "75");
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

  for(int i = 0; i < 4; i++){
    std::cout << fieldCorners[i].x << " " << fieldCorners[i].y << std::endl;
  }

  cv::Mat warpMatrix =  fieldCornersUpdated(fieldCorners, frameSize);

  /*

    ###############################################################################################
    ####################################### CONFIGURE COLORS ######################################
    ###############################################################################################

    def actionConfigureColors():
    	colors = configs.get('colors');
    	selectedColor = 0;
    	colors2 = colors
    	k = None;
    	frame = None;
    	width = None;
    	height = None;
    	resize = 0.5;


    	# Radius of measure Color mean
    	offset = 10


    	def handleMouse(event, x, y, flags, params):
    		global meanColor,x_color,y_color
    		if event == cv2.EVENT_LBUTTONUP:
    			# Get click x y position to a global variable
    			x_color = x
    			y_color = y

    			x1 = int(x - offset); x2 = int(x + offset);
    			y1 = int(y - offset); y2 = int(y + offset);

    			# Check if it exeeds the boundaries
    			if(x1 < 0 or y1 < 0 or  x2 >= frameSize[0] or y2 >= frameSize[1]):
    				print "!!! Exeecds boundaries";
    				return

    			# Get Region of Interest
    			roi = frame[y1:y2, x1:x2];
    			roi = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)

    			#create a round mask
    			mask_pickcolor = np.ones((offset*2,offset*2,1),np.uint8)
    			cv2.circle(mask_pickcolor,(offset,offset),offset,(255),-1)

    			# Blur Image
    			roi = cv2.GaussianBlur(roi, (3, 3), 0)

    			# Find Mean of colors (Excluding outer areas)
    			meanColor = cv2.mean(roi,mask_pickcolor)

    	cv2.namedWindow("pickColors");
    	cv2.setMouseCallback("pickColors", handleMouse);

      	while k != ord("q"):
      		# Read Key and read Video frame
      		k = cv2.waitKey(20) & 0xFF;
      		ret, frame = cap.read()

      		# Save shape for further use
      		height, width = frame.shape[:2];

      		# Change the selected corner on press 1-5 key
      		if(k >= ord("1") and k <= ord("9")):
      			selectedColor = k - ord("1");

      		# Change status text
      		if(selectedColor >= len(colors)):
      			selectedColor = len(colors) - 1;

      		color = colors[selectedColor];
      		status = "Pick color " + str(color[0]);
      		cv2.putText(frame, status, (10,30), font, 0.8, (255, 0, 0), 2);

      		# Get LAB Color of Mouse point
      		color_lab_L = round(meanColor[0])
      		color_lab_A = round(meanColor[1])
      		color_lab_B = round(meanColor[2])
      		color_lab = "["+str(color_lab_L)+","+str(color_lab_A)+","+str(color_lab_B)+"]"

      		cv2.putText(frame, str(color_lab), (10,60), font, 0.8, (55, 255, 55), 2);

      		cv2.circle(frame,(x_color,y_color),offset,(255,255,0),1)

      		if(k == ord("s")):
      			print "funcao de garvar no json"

      		cv2.imshow("pickColors", frame);


      	cv2.destroyWindow("pickColors");

      	#configs.set("fieldCorners", fieldCorners);

      SHOW_DISPLAY = True;

  */

  for(;;){
    /*
    #
    # Check for "pick corner" action
    #
    if (k == ord("f")):
      actionPickCorners();

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

    cv::Mat frame, bin;
    cap >> frame; // get a new frame from camera
    cv::Mat warpedFrame = frame.clone();
    cv::warpPerspective(frame,warpedFrame,warpMatrix,frame.size(),cv::INTER_NEAREST,
                        cv::BORDER_CONSTANT, cv::Scalar() );
    frame = warpedFrame;
    warpedFrame.release();

    colorDetection(frame, bin, 2);

    std::vector<std::vector<cv::Point> > contours;
    std::vector<cv::Vec4i> hierarchy;

    findPos(bin, frame, contours, hierarchy);

    //show results
    //namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
    //Create a window
    cv::namedWindow("Frame", CV_WINDOW_AUTOSIZE);

    //set the callback function for any mouse event
    cv::setMouseCallback("Frame", CallBackFunc, NULL);

    cv::imshow("Frame",frame);

    //cout << frame << ", " << frame << endl;
    int k = cv::waitKey(30);
    if (k == 27 || k == 'q')
      break;
    else if(k == 'f'){
      state = 'f';
      actionPickCorners(cap);
      warpMatrix = fieldCornersUpdated(fieldCorners, frameSize);
      cv::warpPerspective(frame,warpedFrame,warpMatrix,fieldSize,cv::INTER_LINEAR,
                          cv::BORDER_CONSTANT, cv::Scalar() );
    }

  }
  return 0;
}

void changeCameraProp(std::string key, std::string value){
	if (system(NULL)) puts ("Ok");
  else exit (EXIT_FAILURE);
  Json::Value root;
	std::ifstream Config("configs.json");
	Config >> root;
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

void colorDetection(cv::Mat src, cv::Mat &mask, int it){
	cv::Mat hsv, tgt, thrs;
	//3-channel binary mask
	cv::cvtColor(src, hsv, cv::COLOR_BGR2HSV);
	cv::blur(hsv, hsv, cv::Size(3, 3));
	cv::inRange(hsv, cv::Scalar(colorH - rangeH, colorS - rangeS, colorV - rangeV), cv::Scalar(colorH + rangeH + 1 , colorS + rangeS +
	1, colorV + rangeV + 1), mask);

	//image erosion
	cv::Mat element = cv::getStructuringElement( 0,cv::Size( 2,2 ),cv::Point( 0, 0 ) );
  cv::erode( mask, mask, element, cv::Point( 0, 0 ), it	);
	cv::dilate( mask, mask, element, cv::Point( 0, 0 ), 2*it +1);
	cv::erode( mask, mask, element, cv::Point( 0, 0 ), it	);


	//mask aplication
	cv::Mat mask3[] = { mask,mask,mask };
	cv::merge(mask3, 3, thrs);
	cv::bitwise_and(thrs, src, tgt);

	//cv::imshow("Bola", tgt);
	//cv::imshow("HSV", hsv);

}

void findPos(cv::Mat &src,cv::Mat &tgt, std::vector<std::vector<cv::Point> > &contours, std::vector<cv::Vec4i> &hierarchy){
	cv::Mat temp = src.clone();

	if( !temp.empty())
		cv::findContours(temp,contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
		//select best contour
		std::vector<std::vector<cv::Point> > contours_poly( contours.size() );
		std::vector<cv::Point2f> center( contours.size() );
		std::vector<float> radius( contours.size() );

		int realBallRadius = 21, bestBallRadiusDif = 0, final_radius, bestBall = 0;
		cv::Point ball_center;
		std::vector<cv::Point> hull;
		double per, cntArea, relation, radiusDif;

		for( int i = 0; i < contours.size(); i++ ){
			cv::approxPolyDP( cv::Mat(contours[i]), contours_poly[i], 3, true );//verificar se eh necessario
			cv::minEnclosingCircle( (cv::Mat)contours_poly[i], center[i], radius[i] );
			per = arcLength(contours[i], true);
			cv::convexHull(contours[i], hull, true);
			cntArea = cv::contourArea(hull,false) + 0.1;
			relation = (per*radius[i])/2*cntArea;
			radiusDif = abs(realBallRadius - radius[i]);
			if(bestBall == 0 || radiusDif < bestBallRadiusDif){
				final_radius = (int)radius[i];
				ball_center.x = center[i].x;
				ball_center.y = center[i].y;
				bestBall = i;
				bestBallRadiusDif = radiusDif;
			}
		}

		if(bestBall != 0){
			cv::circle( tgt, center[bestBall], (int)radius[bestBall], cv::Scalar(255,0,0), 2, 8, 0 );
		}
}

void CallBackFunc(int event, int x, int y, int flags, void* userdata) {
    switch (state) {
      case 'c':
        if  ( event == cv::EVENT_LBUTTONDOWN )
        {
             x_color = x;
   					 y_color = y;
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

  void actionPickCorners(cv::VideoCapture &cap) {
    cv::namedWindow("pickCorners");
    cv::setMouseCallback("pickCorners", CallBackFunc);
    Json::Value root;
    std::ifstream Config("configs.json");
    Config >> root;
    for(;;) {
          int k = cv::waitKey(30);
          if (k == 27 || k == 'q'){
            state = 'c';
            break;
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

      std::ofstream configs;
      configs.open("configs.json");

      Json::StyledWriter styledWriter;
      configs << styledWriter.write(root);

      configs.close();
  }
