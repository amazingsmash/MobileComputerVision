//
//  ViewController.mm
//  OpenCVExperiment
//
//  Created by Jose Miguel SN on 16/9/16.
//  Copyright © 2016 Jose Miguel SN. All rights reserved.
//

#import "ViewController.h"

#import <opencv2/highgui/cap_ios.h>


@interface ViewController ()

@end

@implementation ViewController

@synthesize videoCamera;
@synthesize imageView;

- (void)viewDidLoad {
  [super viewDidLoad];
  // Do any additional setup after loading the view, typically from a nib.
  
  
  self.videoCamera = [[CvVideoCamera alloc] initWithParentView:imageView];
  self.videoCamera.defaultAVCaptureDevicePosition = AVCaptureDevicePositionBack;
  self.videoCamera.defaultAVCaptureSessionPreset = AVCaptureSessionPreset352x288;
  self.videoCamera.defaultAVCaptureVideoOrientation = AVCaptureVideoOrientationPortrait;
  self.videoCamera.defaultFPS = 30;
  self.videoCamera.grayscaleMode = NO;
  
  self.videoCamera.delegate = self;
}

//CHECK OUT THIS http://docs.opencv.org/2.4/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html
//and this http://study.marearts.com/2013/10/opencv-246-calibration-example-source.html

/*
 //This might not be necessary
 Mat getCameraInternalMatrix(){
 
 //iPhone 6 Back Camera
 
 const double focalLengthMM = 29;
 const double pixelSizeMM = 1.22e-3;
 
 const double f = focalLengthMM / pixelSizeMM; //fx, fy
 //Image size
 int w = 358, h = 288;
 
 cv::Mat1f K = (cv::Mat1f(3, 3) <<
 f, 0, w/2,
 0, f, h/2,
 0, 0,   1);
 
 return K;
 }
 */

- (void)didReceiveMemoryWarning {
  [super didReceiveMemoryWarning];
  // Dispose of any resources that can be recreated.
}

- (IBAction)actionStart:(id)sender;
{
  [self.videoCamera start];
  [self button].hidden = TRUE;
}

#pragma mark - Protocol CvVideoCameraDelegate

enum Operation{
  INVERT,
  SOBEL,
  HOUGH,
  CAMPOS,
  FACADE,
  NOP
} currentOp = NOP;

- (IBAction)changeOperation:(UISegmentedControl *)sender {
  switch ([sender selectedSegmentIndex]){
    case 0:
      currentOp = NOP;
      break;
    case 1:
      currentOp = INVERT;
      break;
    case 2:
      currentOp = SOBEL;
      break;
    case 3:
      currentOp = HOUGH;
      break;
    case 4:
      currentOp = CAMPOS;
      break;
  }
}

void printMat(const Mat& matrix, const std::string& name)
{
  printf("%s:\n", name.c_str());
  for(int j=0; j< matrix.rows; ++j)
  {
    for(int k=0; k< matrix.cols; ++k)
    {
      printf("%lf ", matrix.at<  double >(j,k));
    }
    printf("\n");
  }
}
/*
 void printf3Point( vector< vector< Point3f> > Points, string name)
 {
 FILE * fp;
 fp = fopen(name.c_str() ,"w");
 for(int i=0; i< Points.size(); ++i)
 {
 for(int j=0; j< Points[i].size(); ++j)
 {
 fprintf(fp,"%lf %lf %lf\n", Points[i][j].x, Points[i][j].y, Points[i][j].z);
 }
 fprintf(fp,"\n");
 }
 fclose(fp);
 }
 */

Mat intrinsic_Matrix(3,3, CV_64F); //We will try to refine the previous intrinsic camera matrix
Mat distortion_coeffs(8,1, CV_64F);
bool usingIntrinsicGuess = false;

bool isIntrinsicMatrixValid(const Mat& m){
  if (m.at<double>(0,0) <= 0.0 || m.at<double>(1,1) <= 0.0){
    return false;
  }
  return true;
}

//From UIImage to Mat
- (cv::Mat)cvMatFromUIImage:(UIImage *)image
{
  CGColorSpaceRef colorSpace = CGImageGetColorSpace(image.CGImage);
  CGFloat cols = image.size.width;
  CGFloat rows = image.size.height;
  
  cv::Mat cvMat(rows, cols, CV_8UC4); // 8 bits per component, 4 channels (color channels + alpha)
  
  CGContextRef contextRef = CGBitmapContextCreate(cvMat.data,                 // Pointer to  data
                                                  cols,                       // Width of bitmap
                                                  rows,                       // Height of bitmap
                                                  8,                          // Bits per component
                                                  cvMat.step[0],              // Bytes per row
                                                  colorSpace,                 // Colorspace
                                                  kCGImageAlphaNoneSkipLast |
                                                  kCGBitmapByteOrderDefault); // Bitmap info flags
  
  CGContextDrawImage(contextRef, CGRectMake(0, 0, cols, rows), image.CGImage);
  CGContextRelease(contextRef);
  
  return cvMat;
}

//UIImage to grayscale Mat
- (cv::Mat)cvMatGrayFromUIImage:(UIImage *)image
{
  CGColorSpaceRef colorSpace = CGImageGetColorSpace(image.CGImage);
  CGFloat cols = image.size.width;
  CGFloat rows = image.size.height;
  
  cv::Mat cvMat(rows, cols, CV_8UC1); // 8 bits per component, 1 channels
  
  CGContextRef contextRef = CGBitmapContextCreate(cvMat.data,                 // Pointer to data
                                                  cols,                       // Width of bitmap
                                                  rows,                       // Height of bitmap
                                                  8,                          // Bits per component
                                                  cvMat.step[0],              // Bytes per row
                                                  colorSpace,                 // Colorspace
                                                  kCGImageAlphaNoneSkipLast |
                                                  kCGBitmapByteOrderDefault); // Bitmap info flags
  
  CGContextDrawImage(contextRef, CGRectMake(0, 0, cols, rows), image.CGImage);
  CGContextRelease(contextRef);
  
  return cvMat;
}

-(void) drawPoints:(const std::vector<Point2f>&) points onImage:(Mat&) image{
  for (int i = 0; i < points.size(); ++i){
    printf("Corner: %f, %f\n", points[i].x, points[i].y);
    cv::circle(image, points[i], 20, Scalar(255,0,0));
  }
}

-(vector< Point3f>) objectPointsFromPatternImagePoints: (const vector<Point2f>&) points2D
                                 withPatternObjectSize: (const cv::Size&) patternObjectSize
                                  withPatternImageSize: (const cv::Size&) patternImageSize{
  vector< Point3f> objectPoints;
  for(int j=0; j< points2D.size(); ++j)
  {
    Point2f tImgPT;
    Point3f tObjPT;
    
    double x = points2D[j].x / patternImageSize.width * patternObjectSize.width;
    double y = points2D[j].y / patternImageSize.height * patternObjectSize.height;
    double z = 0;
    
    objectPoints.push_back(Point3f(x,y,z));
  }
  return objectPoints;
}

-(Mat) readGrayJPEGWithName:(NSString*) name{
  NSString* filePath = [[NSBundle mainBundle] pathForResource:name ofType:@"jpg"];
  assert(filePath != nil);
  UIImage* resImage = [UIImage imageWithContentsOfFile:filePath];
  cv::Mat pattern = [self cvMatFromUIImage:resImage];
  cvtColor(pattern, pattern, CV_RGB2GRAY);
  return pattern;
}



-(void) facade:(Mat&) img{
  
  Mat facade = [self readGrayJPEGWithName:@"facade"];
  Mat facadeInScene = [self readGrayJPEGWithName:@"facadeInScene"];
  
  cv::resize(facade, facade, cv::Size(400, 400));
  cv::resize(facadeInScene, facadeInScene, cv::Size(400, 400));
  
  //facade = [self hough:facade];
  //facadeInScene = [self hough:facadeInScene];
  
  
  //Facade
  cv::OrbFeatureDetector featDet;
  vector<cv::KeyPoint> keypointsObj;
  featDet.detect(facade, keypointsObj);
  cv::OrbDescriptorExtractor descExt;
  cv::Mat descObject;
  descExt.compute(facade, keypointsObj, descObject);
  printf("Facade descriptors %d\n", descObject.size[0]);
  
  //Facade in scene
  vector<cv::KeyPoint> keypointsScene;
  featDet.detect(facadeInScene, keypointsScene);
  cv::Mat descScene;
  descExt.compute(facade, keypointsScene, descScene);
  printf("Facade in scene descriptors %d\n", descScene.size[0]);
  
  //Finding matches
  FlannBasedMatcher flann;
  vector<cv::DMatch> matches;
  descObject.convertTo(descObject, CV_32F); //Correct format
  descScene.convertTo(descScene, CV_32F);
  
  //flann.match(descObject, descScene, matches);
  flann.match(descObject, descScene, matches);
  
  //-- Quick calculation of max and min distances between keypoints
  double min_dist = 99999999.0, max_dist = 0.0;
  for( int i = 0; i < descObject.rows; i++ ){
    cv::DMatch match = matches[i];
    double dist = match.distance;
    if( dist < min_dist ) min_dist = dist;
    if( dist > max_dist ) max_dist = dist;
  }
  
  printf("-- Max dist : %f \n", max_dist );
  printf("-- Min dist : %f \n", min_dist );
  
  std::vector< DMatch > good_matches;
  
  for( int i = 0; i < descObject.rows; i++ ){
    if( matches[i].distance < 1.6*min_dist ){
      good_matches.push_back( matches[i]);
    }
  }
  
  
  
  //-- Localize the object
  std::vector<Point2f> obj;
  std::vector<Point2f> scene;
  
  for( int i = 0; i < good_matches.size(); i++ )
  {
    //-- Get the keypoints from the good matches
    obj.push_back( keypointsObj[ good_matches[i].queryIdx ].pt );
    scene.push_back( keypointsScene[ good_matches[i].trainIdx ].pt );
  }
  
  Mat H = findHomography( obj, scene, CV_RANSAC );
  
  std::vector<Point2f> boundsOfFacadeObj;
  boundsOfFacadeObj.push_back(Point2f(0,0));
  boundsOfFacadeObj.push_back(Point2f(399,0));
  
  std::vector<Point2f> boundsOfFacadeScene;
  
  cv::perspectiveTransform(boundsOfFacadeObj, boundsOfFacadeScene, H);
  
  for (int i = 0; i < boundsOfFacadeScene.size(); ++i){
    cv::circle(facadeInScene, boundsOfFacadeScene[i], 20, Scalar(255,0,0));
  }
  img = facadeInScene;
  
  if (true){
    Mat img_matches;
    drawMatches( facade, keypointsObj, facadeInScene, keypointsScene,
                good_matches, img_matches, Scalar::all(-1), Scalar::all(-1),
                vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    img = img_matches;
  }
  
  /*
   std::vector< cv::Point2f > corners;
   int maxCorners = 10;
   double qualityLevel = 0.01;
   double minDistance = 200.0;
   cv::goodFeaturesToTrack(pattern, corners, maxCorners, qualityLevel, minDistance);
   printf("%lu\n", corners.size());
   
   cv::Size imageSize = cv::Size(pattern.cols, pattern.rows);
   
   vector< Point3f> objectPoints = [self objectPointsFromPatterImagePoints:corners
   withPatternObjectSize:cv::Size(10,10) //Facade of 10x10 meters
   withPatternImageSize:imageSize];
   
   cvtColor(pattern, pattern, CV_GRAY2RGB);
   
   vector< Mat> rvecs, tvecs;
   calibrateCamera(objectPoints, corners, imageSize, intrinsic_Matrix, distortion_coeffs, rvecs, tvecs);
   
   [self drawPoints:corners onImage:pattern];
   
   img = pattern;
   */
  
}



-(void) estimateCameraPosition:(Mat&) img{
  
  //Set input params..
  int board_w = 7, board_h = 6;
  float measure=25;
  cv::Size imageSize;
  
  imageSize = cv::Size(img.cols, img.rows);
  Mat gray;
  cvtColor(img, gray, CV_RGB2GRAY);
  vector< Point2f> corners;
  
  bool sCorner = findChessboardCorners(gray, cv::Size(board_w, board_h), corners);
  
  //if find corner success, then
  if(sCorner)
  {
    
    printf("CORNERS FOUND: %lu\n", corners.size());
    if (false){
      drawChessboardCorners(img, cv::Size(board_w, board_h), corners, sCorner);
    }
    
    if(corners.size() == board_w*board_h)
    {
      
      vector< vector< Point2f> > imagePoints;
      vector< vector< Point3f> > objectPoints;
      
      //Storing as vector of 2D image points and 3D world space points
      vector< Point2f> v_tImgPT;
      vector< Point3f> v_tObjPT;
      for(int j=0; j< corners.size(); ++j)
      {
        Point2f tImgPT;
        Point3f tObjPT;
        
        tImgPT.x = corners[j].x;
        tImgPT.y = corners[j].y;
        
        tObjPT.x = j%board_w*measure;
        tObjPT.y = j/board_w*measure;
        tObjPT.z = 0;
        
        v_tImgPT.push_back(tImgPT);
        v_tObjPT.push_back(tObjPT);
      }
      imagePoints.push_back(v_tImgPT);
      objectPoints.push_back(v_tObjPT);
      
      //Refining image points
      if (true){
        cv::cornerSubPix(gray, v_tImgPT, cv::Size(11,11), cv::Size(-1,-1), TermCriteria(TermCriteria::EPS, 30, 0.001));
      }
      
      //Calibrating
      vector< Mat> rvecs, tvecs;
      //Mat intrinsic_Matrix(3,3, CV_64F);
      //Mat distortion_coeffs(8,1, CV_64F);
      
      if (usingIntrinsicGuess && isIntrinsicMatrixValid(intrinsic_Matrix)){
        //printf("NEW FRAME ------------------------\n");
        //printMat(intrinsic_Matrix, "Camera intrinsic matrix");
        
        //calibrateCamera(objectPoints, imagePoints, imageSize, intrinsic_Matrix, distortion_coeffs, rvecs, tvecs, CV_CALIB_USE_INTRINSIC_GUESS | CV_CALIB_FIX_ASPECT_RATIO);
      } else{
        calibrateCamera(objectPoints, imagePoints, imageSize, intrinsic_Matrix, distortion_coeffs, rvecs, tvecs);
        usingIntrinsicGuess = true; //use initial guess does not work
      }
      
      if (rvecs.size() > 1 || tvecs.size() > 1){
        printf("Multiple calibrations??\n");
      }
      
      //Printing calibration
      if (false){
        printf("NEW FRAME ------------------------\n");
        printMat(intrinsic_Matrix, "Camera intrinsic matrix");
        
        for (int i = 0; i < tvecs.size(); ++i){
          printMat(rvecs[i], "Rotation vector");
          printMat(tvecs[i], "Translation vector");
          
          double x = tvecs[i].at<double>(0);
          double y = tvecs[i].at<double>(1);
          double z = tvecs[i].at<double>(2);
          
          printf("Translation length %f\n", sqrt(x*x+y*y+z*z));
        }
      }
      
      //Trying to undistort
      if (false){
        Mat optimalIntrinsicCamMat = getOptimalNewCameraMatrix(intrinsic_Matrix, distortion_coeffs, imageSize, 1.0);
        
        Mat undistortedImage;
        undistort(img, undistortedImage, optimalIntrinsicCamMat, distortion_coeffs);
        img = undistortedImage;
      }
      
      //Trying to draw axis
      if (true){
        Mat rvec(3, 1, CV_64FC1); //rvecs[0];
        Mat tvec(3, 1, CV_64FC1); // tvecs[0];
        
        if (true){
          printf("v_tObjPT: %lu v_tImgPT %lu\n", v_tObjPT.size(), v_tImgPT.size());
          solvePnPRansac(v_tObjPT, v_tImgPT, intrinsic_Matrix, distortion_coeffs, rvec, tvec);
          printMat(rvec, "Rotation vector");
          printMat(tvec, "Translation vector");
        }
        
        //Drawing cube
        vector< Point3f> axis;
        axis.push_back(Point3f(0,0,0));
        axis.push_back(Point3f(3*measure,0,0));
        axis.push_back(Point3f(0,3*measure,0));
        axis.push_back(Point3f(3*measure,3*measure,0));
        
        axis.push_back(Point3f(0,0,-3*measure));
        axis.push_back(Point3f(3*measure,0,-3*measure));
        axis.push_back(Point3f(0,3*measure,-3*measure));
        axis.push_back(Point3f(3*measure,3*measure,-3*measure));
        
        vector<Point2f> imageAxis;
        
        cv::projectPoints(axis, rvec, tvec, intrinsic_Matrix, distortion_coeffs, imageAxis);
        
        cv::line(img, imageAxis[0], imageAxis[1], Scalar(0,0,255), 5);
        cv::line(img, imageAxis[0], imageAxis[2], Scalar(0,0,255), 5);
        cv::line(img, imageAxis[1], imageAxis[3], Scalar(0,0,255), 5);
        cv::line(img, imageAxis[2], imageAxis[3], Scalar(0,0,255), 5);
        
        cv::line(img, imageAxis[0+4], imageAxis[1+4], Scalar(255,0,0), 5);
        cv::line(img, imageAxis[0+4], imageAxis[2+4], Scalar(255,0,0), 5);
        cv::line(img, imageAxis[1+4], imageAxis[3+4], Scalar(255,0,0), 5);
        cv::line(img, imageAxis[2+4], imageAxis[3+4], Scalar(255,0,0), 5);
        
        cv::line(img, imageAxis[0], imageAxis[4], Scalar(0,255,0), 5);
        cv::line(img, imageAxis[1], imageAxis[5], Scalar(0,255,0), 5);
        cv::line(img, imageAxis[2], imageAxis[6], Scalar(0,255,0), 5);
        cv::line(img, imageAxis[3], imageAxis[7], Scalar(0,255,0), 5);
      }
      
    }
    
  } else{
    printf("NO CORNERS FOUND\n");
  }
  
  
}

-(Mat) invertImage:(Mat&) image{
  // Do some OpenCV stuff with the image
  Mat image_copy;
  cvtColor(image, image_copy, CV_BGRA2BGR);
  
  // invert image
  bitwise_not(image_copy, image_copy);
  cvtColor(image_copy, image_copy, CV_BGR2BGRA);
  
  return image_copy;
}

-(void) sobel:(Mat&) image{
  int scale = 1;
  int delta = 0;
  int ddepth = CV_16S;
  
  Mat src_gray;
  //Mat grad;
  
  cv::cvtColor(image, src_gray, cv::COLOR_BGR2GRAY);
  
  /// Generate grad_x and grad_y
  Mat grad_x, grad_y;
  Mat abs_grad_x, abs_grad_y;
  
  /// Gradient X
  //Scharr( src_gray, grad_x, ddepth, 1, 0, scale, delta, BORDER_DEFAULT );
  Sobel( src_gray, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT );
  convertScaleAbs( grad_x, abs_grad_x );
  
  /// Gradient Y
  //Scharr( src_gray, grad_y, ddepth, 0, 1, scale, delta, BORDER_DEFAULT );
  Sobel( src_gray, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT );
  convertScaleAbs( grad_y, abs_grad_y );
  
  /// Total Gradient (approximate)
  addWeighted( abs_grad_x, 0.5, abs_grad_y, 0.5, 0, image );
}

-(Mat) hough:(const Mat&) image{
  Mat dst, cdst;
  Canny(image, dst, 50, 200, 3);
  cvtColor(dst, cdst, CV_GRAY2BGR);
  
#if 0
  vector<Vec2f> lines;
  HoughLines(dst, lines, 1, CV_PI/180, 100, 0, 0 );
  
  for( size_t i = 0; i < lines.size(); i++ )
  {
    float rho = lines[i][0], theta = lines[i][1];
    Point pt1, pt2;
    double a = cos(theta), b = sin(theta);
    double x0 = a*rho, y0 = b*rho;
    pt1.x = cvRound(x0 + 1000*(-b));
    pt1.y = cvRound(y0 + 1000*(a));
    pt2.x = cvRound(x0 - 1000*(-b));
    pt2.y = cvRound(y0 - 1000*(a));
    line( cdst, pt1, pt2, Scalar(0,0,255), 3, CV_AA);
  }
#else
  vector<Vec4i> lines;
  HoughLinesP(dst, lines, 1, CV_PI/180, 50, 50, 10 );
  for( size_t i = 0; i < lines.size(); i++ )
  {
    Vec4i l = lines[i];
    line( cdst, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), Scalar(0,0,255), 3, CV_AA);
  }
#endif
  
  return cdst;
}


#ifdef __cplusplus
- (void)processImage:(Mat&)image;
{
  
  switch(currentOp){
    case INVERT:{
      self.videoCamera.defaultFPS = 30;
      image = [self invertImage:image];
      break;
    }
    case SOBEL:{
      self.videoCamera.defaultFPS = 30;
      [self sobel:image];
      break;
    }
    case HOUGH:{
      self.videoCamera.defaultFPS = 30;
      image = [self hough:image];
      break;
    }
    case CAMPOS:{
      self.videoCamera.defaultFPS = 30;
      [self estimateCameraPosition:image];
      break;
    }
    case FACADE:{
      self.videoCamera.defaultFPS = 30;
      [self facade:image];
      break;
    }
    case NOP:
    default:{
      //NOP
      self.videoCamera.defaultFPS = 30;
    }
  }
  
}



#endif

@end


