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
#include <opencv2/calib3d/calib3d.hpp>
#include <thread>

#include "goal_viewer.h"
class StereoLoc
{
public:
    StereoLoc(std::string config_file_path);
    bool CalcPose(const cv::Mat& left_img, const cv::Mat& right_img);
private:
    bool findCornerSubPix(const cv::Mat& img, std::vector<cv::Point2f>& corners);
    cv::Point3f triangulation(const cv::Point2f& l_p, const cv::Point2f& r_p);
    // TODO: Draw goal in image.
    void drawGoalInImage(const cv::Mat& img);
    void drawLines(const std::vector<cv::Vec2f> lines, cv::Mat& out_img, int line_width);
    bool calcCornersByLine(const cv::Mat& img, std::vector<cv::Point2f>& corners);

    cv::Point2f calcLineInsection(const cv::Vec2f& line1, const cv::Vec2f& line2);

    cv::Point2f calcCentreOfGravity(const cv::Mat& img);
    cv::Ptr<cv::SimpleBlobDetector> blob_detector_;
    cv::SimpleBlobDetector::Params blob_params_;
    cv::Mat left_K_, right_K_;
    cv::Mat left_P_, right_P_;
    float baseline_;
    GoalViewer goal_viewer_;
    std::thread viewer_thread_;
};