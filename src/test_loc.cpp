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
    Mat left_img = imread("/home/cm/Workspaces/stereo_loc/src/stereo_loc/imgs/left_img.png");
    Mat right_img = imread("/home/cm/Workspaces/stereo_loc/src/stereo_loc/imgs/right_img.png");

    StereoLoc sl("/home/cm/Workspaces/stereo_loc/src/stereo_loc/config/gazebo.yaml");
    sl.CalcPose(left_img, right_img);
    waitKey(0);
}