//
//  ViewController.h
//  OpenCVExperiment
//
//  Created by Jose Miguel SN on 16/9/16.
//  Copyright © 2016 Jose Miguel SN. All rights reserved.
//

//#import <UIKit/UIKit.h>


//#import <opencv2/opencv.hpp>

#import <opencv2/highgui/cap_ios.h>
using namespace cv;

@interface ViewController : UIViewController<CvVideoCameraDelegate>{
  CvVideoCamera* videoCamera;
}
@property (nonatomic, retain) CvVideoCamera* videoCamera;

@property (weak, nonatomic) IBOutlet UIImageView *imageView;
@property (weak, nonatomic) IBOutlet UIButton *button;


@end

