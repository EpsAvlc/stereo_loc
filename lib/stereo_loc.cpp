/*
 * Created on Sun Dec 01 2019
 *
 * Copyright (c) 2019 HITSZ-NRSL
 * All rights reserved
 *
 * Author: EpsAvlc
 */

#include "stereo_loc.h"

#include <cmath>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <Eigen/Core>
#include <Eigen/Dense>

using namespace std;
using namespace cv;

StereoLoc::StereoLoc(string config_file_path):goal_viewer_(config_file_path)
{
    FileStorage fs(config_file_path, FileStorage::READ);
    assert(fs.isOpened());
    float left_fx, left_fy, left_cx, left_cy;
    fs["leftCam_fx"] >> left_fx;
    fs["leftCam_fy"] >> left_fy;
    fs["leftCam_cx"] >> left_cx;
    fs["leftCam_cy"] >> left_cy;
    left_K_ = (Mat_<float>(3, 3) << left_fx, 0, left_cx, 0, left_fy, left_cy, 0, 0, 1) ;

    float right_fx, right_fy, right_cx, right_cy;
    fs["rightCam_fx"] >> right_fx;
    fs["rightCam_fy"] >> right_fy;
    fs["rightCam_cx"] >> right_cx;
    fs["rightCam_cy"] >> right_cy;
    right_K_ = (Mat_<float>(3, 3) << right_fx, 0, right_cx, 0, right_fy, right_cy, 0, 0, 1);

    fs["baseline"] >> baseline_;

    left_P_ = (Mat_<float>(3, 4) << left_fx, 0, left_cx, 0, 
                                    0, left_fy, left_cx, 0,
                                    0, 0, 1, 0);

    Mat T = (Mat_<float>(3, 4) << 1, 0, 0, -baseline_, 
                                    0, 1, 0, 0,
                                    0, 0, 1, 0);
    right_P_ = right_K_ * T; 

    blob_params_.minThreshold = 10;
    blob_params_.maxThreshold = 1000;
    blob_params_.filterByArea = true;
    blob_params_.minArea = 100;
    blob_params_.maxArea = 1000000;
    blob_params_.filterByCircularity = true;
    blob_params_.minCircularity = 0.7;
    // blob_params_.
    blob_params_.filterByColor = true;
    blob_params_.blobColor = 0;

    blob_detector_ = SimpleBlobDetector::create(blob_params_);

    /***** Start goal viewer *****/
    viewer_thread_ = thread(&GoalViewer::Run, &goal_viewer_);
}

bool StereoLoc::CalcPose(const cv::Mat& left_img, const cv::Mat& right_img)
{
    vector<Point2f> left_points, right_points;
    if(!findCornerSubPix(left_img, left_points))
    {
        cout << "["<< __FUNCTION__  <<"]:Can't find two corners in left image" << endl;
        return false;
    }
    if(!findCornerSubPix(right_img, right_points))
    {
        cout << "["<< __FUNCTION__  <<"]:Can't find two corners in right image" << endl;
        return false;
    }

    Point3f left_corner = triangulation(left_points[0], right_points[0]);
    Point3f right_corner = triangulation(left_points[1], right_points[1]);

    Eigen::Vector3f t;
    t.x() = left_corner.x;
    t.y() = left_corner.y;
    t.z() = left_corner.z;
    
    float yaw = atan((left_corner.z - right_corner.z) 
    / fabs(left_corner.x - right_corner.x));

    Eigen::Matrix3f R = Eigen::Matrix3f::Zero();
    R(0, 0) = cos(yaw);
    R(0, 2) = sin(yaw);
    R(1, 1) = 1;
    R(2, 1) = -sin(yaw);
    R(2, 2) = cos(yaw);
    Eigen::Matrix3f R_c_w = Eigen::Matrix3f::Zero();
    R_c_w(0, 1) = 1;
    R_c_w(1, 2) = -1;
    R_c_w(2, 0) = 1;
    R = R*R_c_w;
    goal_viewer_.UpdatePose(R, t);
}

bool StereoLoc::findCornerSubPix(const cv::Mat& img, vector<Point2f>& corners)
{
    vector<KeyPoint> key_corners;
    
    blob_detector_->detect(img, key_corners);

    /***** display keypoint *****/
    Mat kp_image;
    drawKeypoints(img, key_corners, kp_image, Scalar(255, 0, 0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    imshow("keypoints", kp_image);

    if(key_corners.size() != 2)
    {
        return false;
    }

    /***** refine keypoints *****/
    const float dilate_ratio = 0.2f;
    for(int i = 0; i < key_corners.size(); i++)
    {
        float roi_size = key_corners[i].size;
        Point2f c = key_corners[i].pt;
        float start_x = max(0.0001f, c.x - roi_size / 2*(1 + dilate_ratio));
        float start_y = max(0.0001f, c.y - roi_size / 2*(1 + dilate_ratio));
        float len_x = min(roi_size * (1+dilate_ratio), img.cols - start_x);
        float len_y = min(roi_size * (1+dilate_ratio), img.rows - start_y);
        Mat roi = img(Rect(start_x, start_y, len_x, len_y));

        cvtColor(roi, roi, COLOR_BGR2GRAY);
        GaussianBlur(roi, roi, Size(3,3), 1);
        Mat bin_roi;
        threshold(roi, bin_roi, 4, 255, THRESH_BINARY_INV);
        // imshow("bin_roi", bin_roi);
        vector<vector<Point>> contours;
        findContours(bin_roi, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
        if(contours.size() != 1)
        {
            cout << "["<< __FUNCTION__  <<"]:find more than one contours in the" << i+1 << "corner image!"<< endl;
            return false;
        }
        RotatedRect min_rect= minAreaRect(contours[0]);
        Point2f final_corner;
        // for(int i = 0; i < contours[0].size(); i++)
        // {
        //     final_corner.x += contours[0][i].x;
        //     final_corner.y += contours[0][i].y;
        // }
        // final_corner.x /= contours[0].size();
        // final_corner.y /= contours[0].size();
        final_corner.x = min_rect.center.x;
        final_corner.y = min_rect.center.y;
        final_corner.x += c.x - roi_size / 2*(1 + dilate_ratio);
        final_corner.y += c.y - roi_size / 2*(1 + dilate_ratio);
        corners.push_back(final_corner);
    }

    // corners.push_back(key_corners[0].pt);
    // corners.push_back(key_corners[1].pt);

    if(corners[0].x > corners[1].x)
    {
        swap(corners[0], corners[1]);
    }

    return true;
}

Point3f StereoLoc::triangulation(const Point2f& l_p, const Point2f& r_p)
{
    Eigen::Vector3f l_x_3, r_x_3;
    l_x_3.x() = (l_p.x - left_K_.at<float>(0, 2)) / left_K_.at<float>(0,0);
    l_x_3.y() = (l_p.y - left_K_.at<float>(1, 2)) / left_K_.at<float>(1,1);
    l_x_3.z() = 1;

    r_x_3.x() = (r_p.x - right_K_.at<float>(0, 2)) / right_K_.at<float>(0,0);
    r_x_3.y() = (r_p.y - right_K_.at<float>(1, 2)) / right_K_.at<float>(1,1);
    r_x_3.z() = 1;

    Eigen::Matrix3f r_x_hat = Eigen::Matrix3f::Zero();
    r_x_hat(0, 1) = -r_x_3.z();
    r_x_hat(0, 2) = r_x_3.y();
    r_x_hat(1, 0) = r_x_3.z();
    r_x_hat(1, 2) = -r_x_3.x();
    r_x_hat(2, 0) = -r_x_3.y();
    r_x_hat(2, 1) = r_x_3.x();

    Eigen::Matrix3f R = Eigen::Matrix3f::Identity();
    Eigen::Vector3f t = Eigen::Vector3f::Zero();
    t.x() = baseline_;

    Eigen::Vector3f A = r_x_hat * R * l_x_3;
    Eigen::Vector3f b = -r_x_hat * t; 
    float s1 = 0;
    s1 = ((A.transpose() * A).inverse() * (A.transpose() * b))(0, 0);
    return Point3f(s1*r_x_3.x(), s1*r_x_3.y(), s1*r_x_3.z());
}