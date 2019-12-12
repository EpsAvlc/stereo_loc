/*
 * Created on Sun Dec 01 2019
 *
 * Copyright (c) 2019 HITSZ-NRSL
 * All rights reserved
 *
 * Author: EpsAvlc
 */

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "stereo_loc.h"

using namespace std;
using namespace cv;
int main()
{
    // Mat left_img = imread("/home/cm/Workspaces/stereo_loc/src/stereo_loc/imgs/left_img.png");
    // Mat right_img = imread("/home/cm/Workspaces/stereo_loc/src/stereo_loc/imgs/right_img.png");

    // StereoLoc sl("/home/cm/Workspaces/stereo_loc/src/stereo_loc/config/gazebo.yaml");
    // sl.CalcPose(left_img, right_img);

    Mat left_img = imread("/home/cm/OneDrive/课程/计算机视觉/cv_project/Cv_Project3_Photos/left cam/4-1.bmp");
    Mat right_img = imread("/home/cm/OneDrive/课程/计算机视觉/cv_project/Cv_Project3_Photos/right cam/4-1.bmp");

    StereoLoc sl("/home/cm/Workspaces/stereo_loc/src/stereo_loc/config/real.yaml");
    sl.CalcPose(left_img, right_img);
    // imshow("left_img", left_img);

    waitKey(0);
    // while(1);
}
