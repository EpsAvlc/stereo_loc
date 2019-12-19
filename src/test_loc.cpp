/*
 * Created on Sun Dec 01 2019
 *
 * Copyright (c) 2019 HITSZ-NRSL
 * All rights reserved
 *
 * Author: EpsAvlc
 */

#include <ros/ros.h>

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "stereo_loc.h"

using namespace std;
using namespace cv;
int main(int argc, char** argv)
{
    ros::init(argc, argv, "test_loc");
    ros::NodeHandle nh("~");
    string lhs_img_str, rhs_img_str, config_str;
    nh.param<string>("left_cam_img", lhs_img_str, "abc");
    nh.param<string>("right_cam_img", rhs_img_str, "def");
    nh.param<string>("config_file", config_str, "def");

    Mat left_img = imread(lhs_img_str);
    Mat right_img = imread(rhs_img_str);

    StereoLoc sl(config_str);
    sl.CalcPose(left_img, right_img);


    waitKey(0);
    // while(1);
}
