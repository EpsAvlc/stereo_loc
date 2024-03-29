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

#include <opencv2/core/eigen.hpp>

#include "math_tools.h"

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

    /***** For drawing *****/
    vector<float> vec_t;
    fs["t"] >> vec_t;
    for(int i = 0; i < 3; i++)
        t_(i) = vec_t[i];
    vector<float> vec_R;
    fs["R"] >> vec_R;
    for(int i = 0; i < 9; i++)
        R_(i) = vec_R[i];

    fs["is_sim"] >> is_sim_;
    fs["goal_height"] >> goal_height_;
    fs["goal_width1"] >> goal_width1_;
    fs["goal_width2"] >> goal_width2_;
    fs["goal_length"] >> goal_length_;

    /***** For blob detection *****/
    fs["blob_minThres"] >> blob_minThres_;
    fs["blob_maxThres"] >> blob_maxThres_;
    fs["blob_minArea"] >> blob_minArea_;
    fs["blob_maxArea"] >> blob_maxArea_;
    fs["blob_minCircularity"] >> blob_minCircularity_;
    fs["blob_minInertiaRatio"] >> blob_minInertiaRatio_;
    fs["blob_minConvexity"] >> blob_minConvexity_;
    fs["Canny_lowThres"] >> Canny_lowThres_;
    fs["Canny_highThres"] >> Canny_highThres_;
    fs["line_roi_size"] >> line_roi_size_;
    fs["Hough_minLength"] >> Hough_minLength_;
    fs["keypoint_thres"] >> keypoint_thres_;
 
    /***** Init blob parameters *****/
    blob_params_.minThreshold = blob_minThres_;
    blob_params_.maxThreshold = blob_maxThres_;
    blob_params_.filterByArea = true;
    blob_params_.minArea = blob_minArea_;
    blob_params_.maxArea = blob_maxArea_;
    blob_params_.filterByCircularity = true;
    blob_params_.minCircularity = blob_minCircularity_;
    blob_params_.maxCircularity = 1;
    blob_params_.filterByColor = true;
    blob_params_.blobColor = 0;

    blob_params_.filterByConvexity = true;
    blob_params_.minConvexity = blob_minConvexity_;
    blob_params_.maxConvexity = 1;

    blob_params_.filterByInertia = true;
    blob_params_.minInertiaRatio = blob_minInertiaRatio_;
    blob_params_.maxInertiaRatio = 1;

    // blob_params_.thresholdStep = 3;

    blob_detector_ = SimpleBlobDetector::create(blob_params_);

    /***** Start goal viewer *****/
    viewer_thread_ = thread(&GoalViewer::Run, &goal_viewer_);
}

bool StereoLoc::CalcPose(const cv::Mat& left_img, const cv::Mat& right_img)
{
    vector<Point2f> left_corners_2d, right_corners_2d;
    if(!findCornerSubPix(left_img, left_corners_2d))
    {
        cout << "["<< __FUNCTION__  <<"]:Can't find two corners in left image" << endl;
        return false;
    }

    if(!findCornerSubPix(right_img, right_corners_2d))
    {
        cout << "["<< __FUNCTION__  <<"]:Can't find two corners in right image" << endl;
        return false;
    }

    Point3f left_corner_3d = triangulation(left_corners_2d[0], right_corners_2d[0]);
    Point3f right_corner_3d = triangulation(left_corners_2d[1], right_corners_2d[1]);

    // cout << "left: " << endl;
    // cout << left_corners_2d[0] << endl;
    // cout << right_corners_2d[0] << endl;
    // cout << "right: " << endl;
    // cout << left_corners_2d[1] << endl;
    // cout << right_corners_2d[1] << endl;

    cout << left_corner_3d << endl;
    cout << right_corner_3d << endl;
    cout << sqrt((left_corner_3d.x - right_corner_3d.x) * (left_corner_3d.x - right_corner_3d.x) + 
    (left_corner_3d.y - right_corner_3d.y) * (left_corner_3d.y - right_corner_3d.y)
    +(left_corner_3d.z - right_corner_3d.z) * (left_corner_3d.z - right_corner_3d.z)) << endl;

    /***** line method ******/
    Eigen::MatrixXf left_P(3, 4); 
    left_P = Eigen::MatrixXf::Zero(3, 4);
    left_P.block(0, 0, 3, 3) = Eigen::Matrix3f::Identity();
    Eigen::Matrix3f left_K_eigen;
    cv2eigen(left_K_, left_K_eigen);
    left_P = left_K_eigen * left_P;
    bool left_successed = calcCornersByLine(left_img, left_corner_3d, right_corner_3d, left_P, left_corners_2d);

    Eigen::MatrixXf right_P(3, 4);
    right_P = Eigen::MatrixXf::Zero(3, 4);
    right_P.block(0, 0, 3, 3) = R_;
    right_P.block(0, 3, 3, 1) = t_;
    Eigen::Matrix3f right_K_eigen;
    cv2eigen(right_K_, right_K_eigen);
    right_P = right_K_eigen * right_P;
    bool right_successed = calcCornersByLine(right_img, left_corner_3d, right_corner_3d, right_P, right_corners_2d);

    if(!(right_successed && left_successed))
        return false;

    left_corner_3d = triangulation(left_corners_2d[0], right_corners_2d[0]);
    right_corner_3d = triangulation(left_corners_2d[1], right_corners_2d[1]);

    // cout << left_corner_3d << endl;
    // cout << right_corner_3d << endl;
    // cout << sqrt((left_corner_3d.x - right_corner_3d.x) * (left_corner_3d.x - right_corner_3d.x) + 
    // (left_corner_3d.y - right_corner_3d.y) * (left_corner_3d.y - right_corner_3d.y)
    // +(left_corner_3d.z - right_corner_3d.z) * (left_corner_3d.z - right_corner_3d.z)) << endl;

    // /***** for viewer *****/
    Eigen::Vector3f t;
    t.x() = left_corner_3d.x;
    t.y() = left_corner_3d.y;
    t.z() = left_corner_3d.z;
    
    float pitch = atan((right_corner_3d.z - left_corner_3d.z) 
    / (right_corner_3d.x - left_corner_3d.x));

    Eigen::Matrix3f R = Eigen::Matrix3f::Zero();
    R(0, 0) = cos(-pitch);
    R(0, 2) = sin(-pitch);
    R(1, 1) = 1;
    R(2, 1) = -sin(-pitch);
    R(2, 2) = cos(-pitch);
    Eigen::Matrix3f R_c_w = Eigen::Matrix3f::Zero();
    R_c_w(0, 1) = 1;
    R_c_w(1, 2) = -1;
    R_c_w(2, 0) = 1;
    R = R*R_c_w;

    Mat left_img_clone = left_img.clone();
    drawGoal(left_img_clone, left_corner_3d, right_corner_3d, Scalar(0, 0, 255));

    resize(left_img_clone, left_img_clone, Size(), 0.5, 0.5);
    imshow("goal", left_img_clone);
    goal_viewer_.UpdatePose(R, t);
}

bool StereoLoc::findCornerSubPix(const cv::Mat& img, vector<Point2f>& corners)
{
    vector<KeyPoint> key_corners;
    
    blob_detector_->detect(img, key_corners);

    Mat kp_image;
    drawKeypoints(img, key_corners, kp_image, Scalar(0, 0, 255),
    DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    judgeCorners(key_corners, img);

    // drawKeypoints(kp_image, key_corners, kp_image, Scalar(255, 0, 0),
    // DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    // resize(kp_image, kp_image, Size(), 0.5, 0.5);
    // imshow("keypoints", kp_image);
    // waitKey(0);

    /***** refine keypoints *****/
    const float dilate_ratio = 0.35f;
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
        // GaussianBlur(roi, roi, Size(3,3), 1);

        Mat bin_roi;
        threshold(roi, bin_roi, 4, 255, THRESH_BINARY_INV|THRESH_OTSU);
        vector<vector<Point>> contours;
        findContours(bin_roi, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
        float max_area = 0;
        float max_area_index;
        for(int i = 0; i < contours.size(); i++)
        {
            if(contours[i].size() < 5)
                continue;
            RotatedRect rr = fitEllipse(contours[i]);
            if(checkEllipseShape(bin_roi, contours[i], rr) == false)
            {
                if(rr.size.area() > max_area)
                {
                    max_area = rr.size.area();
                    max_area_index = i;
                }
            }
   
        }
        Mat contour_mat(bin_roi.size(), CV_8UC1, Scalar(0));
        drawContours(contour_mat, contours, max_area_index, Scalar(255), -1);

        // imshow("contour_mat", contour_mat);
        // imshow("bin_roi", bin_roi);
 

        // Point2f final_corner = calcCentreOfGravity(contour_mat);
        // circle(contour_mat, final_corner, 2, Scalar(0), -1);

        Point2f final_corner = fitEllipse(contours[max_area_index]).center;
        // imshow("contour_mat", contour_mat);
        // waitKey(0);
        final_corner.x += c.x - roi_size / 2*(1 + dilate_ratio);
        final_corner.y += c.y - roi_size / 2*(1 + dilate_ratio);
        corners.push_back(final_corner);

        // cout << final_corner << endl;
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

    // Eigen::Matrix3f R = Eigen::Matrix3f::Identity();
    // Eigen::Vector3f t = Eigen::Vector3f::Zero();
    // t.x() = baseline_;

    Eigen::Vector3f A = r_x_hat * R_ * l_x_3;
    Eigen::Vector3f b = -r_x_hat * t_; 
    float s2 = ((A.transpose() * A).inverse() * (A.transpose() * b))(0, 0);
    A = R_ * l_x_3;
    b = s2*r_x_3 - t_;
    float s1 = ((A.transpose() * A).inverse() * (A.transpose() * b))(0, 0);
    return Point3f(s1*l_x_3.x(), s1*l_x_3.y(), s1*l_x_3.z());
}

Point2f StereoLoc::calcCentreOfGravity(const cv::Mat& img)
{
    float moment_00 = 0, moment_10 = 0, moment_01 = 0;
    for(int i = 0; i < img.cols; i++)
        for(int j = 0; j < img.rows; j++)
        {
            int val = img.at<uchar>(j, i);
            if(val > 0)
            {
                float weight = val;
                moment_00 += weight;
                moment_10 += j * weight;
                moment_01 += i * weight;
            }
        }
    return Point2f(moment_01 / moment_00, moment_10 / moment_00);
}

bool StereoLoc::calcCornersByLine(const Mat& img, const Point3f& left_corner_3d, const Point3f& right_corner_3d, const Eigen::MatrixXf& P, vector<Point2f>& refine_corners_2d)
{
    // cout <<" enter " << endl;
    Mat gray_img;

    cvtColor(img, gray_img, COLOR_BGR2GRAY);
    // equalizeHist(gray_img, gray_img);
    Mat bin_img;
    Mat img_no_net;
    Mat img_no_net_eq;


    if(!is_sim_)
    {
        removeNet(img, img_no_net);
        // equalizeHist(img_no_net, img_no_net_eq);
        // GaussianBlur(img_no_net_eq, img_no_net_eq, Size(5, 5), 0);
        // Mat thres_img;
        // threshold(img_no_net_eq, thres_img, 30, 255, THRESH_BINARY|THRESH_OTSU);
        // Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
        // Mat bin_img_closed;
        // morphologyEx(thres_img, thres_img, MORPH_OPEN, element);
        // imshow("Thres", thres_img);

        Canny(img_no_net, bin_img, Canny_lowThres_, Canny_highThres_);

    }
    else
    {
        Canny(gray_img, bin_img, Canny_lowThres_, Canny_highThres_);
    }
    vector<vector<Point>> Canny_contours;

    findContours(bin_img, Canny_contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    Mat Canny_Contour_mat(bin_img.size(), CV_8UC1, Scalar(0));
    for(int i = 0; i < Canny_contours.size(); i++)
    {
        RotatedRect rr = minAreaRect(Canny_contours[i]);
        if(rr.size.area() < 250)
            continue;
        drawContours(Canny_Contour_mat, Canny_contours, i, Scalar(255));
    }

    Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(Canny_Contour_mat, Canny_Contour_mat, MORPH_CLOSE, element);

    Mat Canny_Contour_mat_disp = Canny_Contour_mat.clone();
    resize(Canny_Contour_mat_disp, Canny_Contour_mat_disp, Size(), 0.5, 0.5);
    // if(abs(P(0, 3)) < 1e-5)
    //     imshow("canny_left", Canny_Contour_mat_disp);
    // else
    //     imshow("canny_right", Canny_Contour_mat_disp);

    /***** construct mask *****/
    Eigen::MatrixXf pts_4d(4, 6);

    pts_4d(0, 0) = left_corner_3d.x;
    pts_4d(1, 0) = left_corner_3d.y + 0.6;
    pts_4d(2, 0) = left_corner_3d.z;
    pts_4d(3, 0) = 1;
    pts_4d(0, 1) = left_corner_3d.x;
    pts_4d(1, 1) = left_corner_3d.y + goal_height_ - 0.4;
    pts_4d(2, 1) = left_corner_3d.z;
    pts_4d(3, 1) = 1;
    pts_4d(0, 2) = right_corner_3d.x;
    pts_4d(1, 2) = right_corner_3d.y + 0.6;
    pts_4d(2, 2) = right_corner_3d.z;
    pts_4d(3, 2) = 1;
    pts_4d(0, 3) = right_corner_3d.x;
    pts_4d(1, 3) = right_corner_3d.y + goal_height_ - 0.4;
    pts_4d(2, 3) = right_corner_3d.z;
    pts_4d(3, 3) = 1;
    pts_4d(0, 4) = left_corner_3d.x + 0.5;
    pts_4d(1, 4) = left_corner_3d.y - 0.15;
    pts_4d(2, 4) = left_corner_3d.z;
    pts_4d(3, 4) = 1;
    pts_4d(0, 5) = right_corner_3d.x + 0.5;
    pts_4d(1, 5) = right_corner_3d.y - 0.15;
    pts_4d(2, 5) = right_corner_3d.z;
    pts_4d(3, 5) = 1;

    Eigen::MatrixXf pts_homo_2d = P * pts_4d;
    vector<Point2f> pts_2d;
    for(int i = 0; i < pts_4d.cols(); i++)
    {
        Point2f cur_pt;
        cur_pt.x = pts_homo_2d(0, i) / pts_homo_2d(2, i); 
        cur_pt.y = pts_homo_2d(1, i) / pts_homo_2d(2, i); 
        pts_2d.push_back(cur_pt);
    }

    Mat left_pillar_mask(img.size(), CV_8UC1, Scalar(0));
    line(left_pillar_mask, pts_2d[0] , pts_2d[1], Scalar(255), line_roi_size_);
    Mat right_pillar_mask(img.size(), CV_8UC1, Scalar(0));
    line(right_pillar_mask, pts_2d[2] , pts_2d[3], Scalar(255), line_roi_size_);
    Mat horizon_pillar_mask(img.size(), CV_8UC1, Scalar(0));
    line(horizon_pillar_mask, pts_2d[4] , pts_2d[5], Scalar(255), line_roi_size_);



    Mat left_pillor_roi, right_pillor_roi, horizon_pillar_roi;
    Canny_Contour_mat.copyTo(left_pillor_roi, left_pillar_mask);
    Canny_Contour_mat.copyTo(right_pillor_roi, right_pillar_mask);
    Canny_Contour_mat.copyTo(horizon_pillar_roi, horizon_pillar_mask);
    /***** disp_roi ******/
    Mat left_pillor_roi_disp;
    resize(left_pillor_roi, left_pillor_roi_disp, Size(), 0.5, 0.5);
    // imshow("left_pillor_roi_disp", left_pillor_roi_disp);

    Mat gray_img_disp = gray_img.clone();
    cvtColor(gray_img_disp, gray_img_disp, COLOR_GRAY2BGR);
    /***** left_pillor *****/
    vector<pair<Vec2f, int>> left_lines;
    vector<vector<Point>> contours;
    findContours(left_pillor_roi, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    for(int i = 0; i < contours.size(); i++)
    {
        if(contours[i].size() < 200)
            continue;
        Vec2f cur_line;
        int vote_num = fitLineRansac(contours[i], cur_line);
        // if((cur_line[1] <= 0.03) || (CV_PI - cur_line[1] <= 0.03))
        // {
            bool redundant = false;
            for(int i = 0; i < left_lines.size(); i++)
            {
                if(fabs(left_lines[i].first[0] - cur_line[0]) < 10 && fabs(left_lines[i].first[1] - cur_line[1] < 0.002))
                {
                    redundant = true;
                    break;
                }
                if(hasInsection(left_lines[i].first, cur_line, img.rows))
                {
                    redundant = true;
                    break;
                }
            }
            if(! redundant)
                left_lines.push_back(make_pair(cur_line, vote_num));
            // Mat contour_mat(left_pillar_mask.size(), CV_8UC3, Scalar(0, 0, 0));
            // drawContours(contour_mat, contours, i, Scalar(255, 255,255));
            // drawLine(cur_line, contour_mat, 1, Scalar(0, 0, 255));
            // imshow("lines", contour_mat);
            // waitKey(0);
        // }
    }

    if(left_lines.size() > 2)
    {
        sort(left_lines.begin(), left_lines.end(), 
        [](pair<Vec2f, int>& lhs, pair<Vec2f, int>& rhs)
        {
            return lhs.second > rhs.second;
        });
    }
    
    drawLine(left_lines[0].first, gray_img_disp, 1, Scalar(0, 0, 255));
    drawLine(left_lines[1].first, gray_img_disp, 1, Scalar(0, 0, 255));


    cout << left_lines[0].first << endl;
    cout << left_lines[1].first << endl;

    /***** right pillor *****/
    findContours(right_pillor_roi, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    vector<pair<Vec2f, int>> right_lines;
    for(int i = 0; i < contours.size(); i++)
    {
        if(contours[i].size() < 150)
            continue;
        Vec2f cur_line;
        int vote_num = fitLineRansac(contours[i], cur_line);
        bool redundant = false;
        for(int i = 0; i < right_lines.size(); i++)
        {
            if(fabs(right_lines[i].first[0] - cur_line[0]) < 10 && fabs(right_lines[i].first[1] - cur_line[1] < 0.002))
            {
                redundant = true;
                if(vote_num > right_lines[i].second)
                {
                    // right_lines[i] = make_pair(cur_line, vote_num);
                }
                break;
            }
        }
        if(! redundant)
        {
            right_lines.push_back(make_pair(cur_line, vote_num));
        }
    }
    if(right_lines.size() > 2)
    {
        sort(right_lines.begin(), right_lines.end(), 
        [](pair<Vec2f, int>& lhs, pair<Vec2f, int>& rhs)
        {
            return lhs.second > rhs.second;
        });
    }
    drawLine(right_lines[0].first, gray_img_disp, 1, Scalar(0, 0, 255));
    drawLine(right_lines[1].first, gray_img_disp, 1, Scalar(0, 0, 255));
    // cout << right_lines[0].first << endl;
    // cout << right_lines[1].first << endl;
    /***** Horizon Pillar *****/
    findContours(horizon_pillar_roi, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    vector<pair<Vec2f, int>> horizon_lines;
    for(int i = 0; i < contours.size(); i++)
    {
        if(contours[i].size() < 350)
            continue;
        Vec2f cur_line;
        int vote_num = fitLineRansac(contours[i], cur_line);
        horizon_lines.push_back(make_pair(cur_line, vote_num));
    }
    if(horizon_lines.size() > 2)
    {
        sort(horizon_lines.begin(), horizon_lines.end(), 
        [](pair<Vec2f, int>& lhs, pair<Vec2f, int>& rhs)
        {
            return lhs.second > rhs.second;
        });
        if(horizon_lines[0].first[0] > horizon_lines[1].first[0])
        {
            swap(horizon_lines[0], horizon_lines[1]);
        }
    }


    drawLine(horizon_lines[0].first, gray_img_disp, 1, Scalar(0, 0, 255));
    // drawLine(horizon_lines[1].first, gray_img_disp, 1, Scalar(0, 0, 255));


    // // /***** Display hough lines *****/


    // /***** for vertical lines *****/

    // vector<Vec2f> vertical_lines;
    // for(int i = 0; i < lines.size(); i++)
    // {
    //     float rho = lines[i][0], theta = lines[i][1];
    //     if((theta <= 0.03) || (CV_PI - theta <= 0.03))
    //     {
    //         vertical_lines.push_back(lines[i]);
    //     }
    // }

    // if(vertical_lines.size() < 4)
    // {
    //     cout << "There are not enough vertical lines. " << endl; 
    //     return false;
    // }
  
    // // cout << vertical_lines.size() << endl;

    // sort(vertical_lines.begin(), vertical_lines.end(), [](Vec2f& lhs, Vec2f& rhs)
    // {   
    //     return lhs[0] < rhs[0];
    // });


    // judgeVerticalLines(vertical_lines, img_no_net_eq);
    // drawLine(vertical_lines[0], gray_img_disp, 2, Scalar(255, 0, 0));
    // drawLine(vertical_lines[1], gray_img_disp, 2, Scalar(0, 255, 0));
    // drawLine(vertical_lines[2], gray_img_disp, 2, Scalar(0, 0, 255));
    // drawLine(vertical_lines[3], gray_img_disp, 2, Scalar(255, 255, 0));

    // // drawLines(lines, gray_img_disp, 1, Scalar(0, 0, 255));

    // /***** for horizon lines *****/
    // vector<Vec2f> horizon_lines;
    // for(int i = 0; i < lines.size(); i++)
    // {
    //     float rho = lines[i][0], theta = lines[i][1];

    //     float kpt_theta = atan2(-(pts_2d[0].x - pts_2d[1].x), (pts_2d[0].y - pts_2d[1].y + 0.000001));
    
    //     float k = -1/(tan(theta) + 0.0000001);
    //     if(fabs(k) < CV_PI/180 * 20)
    //         horizon_lines.push_back(lines[i]);
    // }
    // drawLines(horizon_lines, gray_img_disp, 1, Scalar(0, 0, 255));

    // judgeHorizonLines(horizon_lines, img_no_net_eq);
    
    Vec2f horizon_pillar_line = horizon_lines[0].first;
    
    Vec2f left_pillar_line, right_pillar_line;
    left_pillar_line = (left_lines[0].first + left_lines[1].first) / 2;
    right_pillar_line = (right_lines[0].first + right_lines[1].first) / 2;

    Point2f left_point = calcLineInsection(left_pillar_line, horizon_pillar_line);

    Point2f right_point = calcLineInsection(right_pillar_line, horizon_pillar_line);

    if(left_point.x > right_point.x)
    {
        swap(left_point, right_point);
    }
    putText(gray_img_disp, "left", left_point, FONT_HERSHEY_PLAIN, 3, Scalar(128), 2);
    circle(gray_img_disp, left_point, 5, Scalar(0), -1);
    putText(gray_img_disp, "right", right_point, FONT_HERSHEY_PLAIN, 3, Scalar(128), 2);
    circle(gray_img_disp, right_point, 5, Scalar(0), -1);

    resize(gray_img_disp, gray_img_disp, Size(), 0.5, 0.5);
    // cout << P << endl;
    if(abs(P(0, 3)) < 1e-5)
        imshow("gray_img_left", gray_img_disp);
    else
        imshow("gray_img_right", gray_img_disp);

    refine_corners_2d[0] = left_point;
    refine_corners_2d[1] = right_point;

    return true;
}

void StereoLoc::drawLines(const vector<Vec2f>& lines, Mat& out_img, int line_width, const Scalar& color)
{
    float line_length = sqrt(out_img.cols * out_img.cols + out_img.rows * out_img.rows);
    for( size_t i = 0; i < lines.size(); i++ )
	{
		float rho = lines[i][0], theta = lines[i][1];
        // cout << theta << endl;
		Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a*rho, y0 = b*rho;
		pt1.x = cvRound(x0 + line_length*(-b));
		pt1.y = cvRound(y0 + line_length*(a));
		pt2.x = cvRound(x0 - line_length*(-b));
		pt2.y = cvRound(y0 - line_length*(a));
		line(out_img, pt1, pt2, color, line_width, CV_AA);
	}
}

void StereoLoc::drawLine(const Vec2f& l, Mat& out_img, int line_width, const Scalar& color)
{
    float line_length = sqrt(out_img.cols * out_img.cols + out_img.rows * out_img.rows);

    float rho = l[0], theta = l[1];
    // cout << theta << endl;
    Point pt1, pt2;
    double a = cos(theta), b = sin(theta);
    double x0 = a*rho, y0 = b*rho;
    pt1.x = cvRound(x0 + line_length*(-b));
    pt1.y = cvRound(y0 + line_length*(a));
    pt2.x = cvRound(x0 - line_length*(-b));
    pt2.y = cvRound(y0 - line_length*(a));
    line(out_img, pt1, pt2, color, line_width, CV_AA);
}

Point2f StereoLoc::calcLineInsection(const Vec2f& line1, const Vec2f& line2)
{
    /***** points for line1 *****/
    Point2f pt1, pt2;
    double a = cos(line1[1]), b = sin(line1[1]);
    double x0 = a*line1[0], y0 = b*line1[0];
    pt1.x = cvRound(x0 + 1000*(-b));
    pt1.y = cvRound(y0 + 1000*(a));
    pt2.x = cvRound(x0 - 1000*(-b));
    pt2.y = cvRound(y0 - 1000*(a));

    /***** points for line2 *****/
    Point2f pt3, pt4;
    a = cos(line2[1]);
    b = sin(line2[1]);
    x0 = a*line2[0];
    y0 = b*line2[0];
    pt3.x = cvRound(x0 + 1000*(-b));
    pt3.y = cvRound(y0 + 1000*(a));
    pt4.x = cvRound(x0 - 1000*(-b));
    pt4.y = cvRound(y0 - 1000*(a));

	float x, y;
	float X1 = pt1.x - pt2.x, Y1 = pt1.y - pt2.y, X2 = pt3.x - pt4.x, Y2 = pt3.y - pt4.y;

	if (X1*Y2 == X2*Y1)
        return cv::Point((pt2.x + pt3.x) / 2, (pt2.y + pt3.y) / 2);
 
	float A = X1*pt1.y - Y1*pt1.x, B = X2*pt3.y - Y2*pt3.x;

	y = (A*Y2 - B*Y1) / (X1*Y2 - X2*Y1);
	x = (B*X1 - A*X2) / (Y1*X2 - Y2*X1);
	return cv::Point2f(x, y);
}

void StereoLoc::drawGoal(Mat& img, const Point3f& left_corner, const Point3f& right_corner, const Scalar& color)
{
    float pitch = atan((right_corner.z - left_corner.z) 
    / (right_corner.x - left_corner.x));

    float roll = 0.01;

    // left front up, right front up, left front down, right front down...
    Eigen::MatrixXf pts_3d(3, 12);
    pts_3d(0, 0) = left_corner.x;
    pts_3d(1, 0) = left_corner.y;
    pts_3d(2, 0) = left_corner.z;
    pts_3d(0, 1) = right_corner.x;
    pts_3d(1, 1) = right_corner.y;
    pts_3d(2, 1) = right_corner.z;
    pts_3d(0, 2) = left_corner.x;
    pts_3d(1, 2) = left_corner.y + goal_height_;
    pts_3d(2, 2) = left_corner.z;
    pts_3d(0, 3) = pts_3d(0, 1);
    pts_3d(1, 3) = pts_3d(1, 1) + goal_height_;
    pts_3d(2, 3) = pts_3d(2, 1);
    
    pts_3d(0, 4) = pts_3d(0, 0) - goal_width2_ * sin(pitch);
    pts_3d(1, 4) = pts_3d(1, 0) + goal_width2_ * sin(roll);
    pts_3d(2, 4) = pts_3d(2, 0) + goal_width2_ * cos(pitch) * cos(roll);    
    pts_3d(0, 5) = pts_3d(0, 1) - goal_width2_ * sin(pitch);
    pts_3d(1, 5) = pts_3d(1, 1) + goal_width2_ * sin(roll);
    pts_3d(2, 5) = pts_3d(2, 1) + goal_width2_ * cos(pitch) * cos(roll);    
    pts_3d(0, 6) = pts_3d(0, 2) - goal_width1_ * sin(pitch);
    pts_3d(1, 6) = pts_3d(1, 2) + goal_width1_ * sin(roll);
    pts_3d(2, 6) = pts_3d(2, 2) + goal_width1_ * cos(pitch) * cos(roll);
    pts_3d(0, 7) = pts_3d(0, 3) - goal_width1_ * sin(pitch);
    pts_3d(1, 7) = pts_3d(1, 3) + goal_width1_ * sin(roll);
    pts_3d(2, 7) = pts_3d(2, 3) + goal_width1_ * cos(pitch) * cos(roll);
    pts_3d(0, 8) = (left_corner.x + right_corner.x) / 2;
    pts_3d(1, 8) = left_corner.y + goal_height_ / 2;
    pts_3d(2, 8) = left_corner.z;
    pts_3d(0, 9) = (left_corner.x + right_corner.x) / 2;
    pts_3d(1, 9) = left_corner.y + goal_height_ / 2 - goal_height_ / 4;
    pts_3d(2, 9) = left_corner.z;
    pts_3d(0, 10) = (left_corner.x + right_corner.x) / 2 + goal_height_ / 4;
    pts_3d(1, 10) = left_corner.y + goal_height_ / 2;
    pts_3d(2, 10) = left_corner.z;
    pts_3d(0, 11) = (left_corner.x + right_corner.x) / 2;
    pts_3d(1, 11) = left_corner.y + goal_height_ / 2;
    pts_3d(2, 11) = left_corner.z - goal_height_ / 4;;

    Eigen::Matrix3f left_K_eigen;
    cv2eigen(left_K_, left_K_eigen);
    Eigen::MatrixXf pts_2d_eigen = left_K_eigen * pts_3d;

    Point2f pts_2d[12];
    for(int i = 0; i < pts_2d_eigen.cols(); i++)
    {
        pts_2d[i].x = pts_2d_eigen(0, i) / pts_2d_eigen(2, i);
        pts_2d[i].y = pts_2d_eigen(1, i) / pts_2d_eigen(2, i); 
    }

    line(img, pts_2d[0], pts_2d[1], color, 7);
    line(img, pts_2d[0], pts_2d[2], color, 7);
    line(img, pts_2d[1], pts_2d[3], color, 7);
    line(img, pts_2d[0], pts_2d[4], color, 3);
    // line(img, pts_2d[5], pts_2d[4], color, 3);
    line(img, pts_2d[6], pts_2d[4], color, 3);
    line(img, pts_2d[7], pts_2d[5], color, 3);
    line(img, pts_2d[1], pts_2d[5], color, 3);
    line(img, pts_2d[2], pts_2d[6], color, 3);
    line(img, pts_2d[3], pts_2d[7], color, 3);
    line(img, pts_2d[6], pts_2d[7], color, 3);

    drawArrow(img, pts_2d[8], pts_2d[9], 17, 15, Scalar(255, 0, 0), 2);
    drawArrow(img, pts_2d[8], pts_2d[10], 17, 15, Scalar(0, 255, 0), 2);
    drawArrow(img, pts_2d[8], pts_2d[11], 17, 15, Scalar(255, 0, 255), 2);
}

void StereoLoc::drawArrow(cv::Mat& img, cv::Point pStart, cv::Point pEnd, int len, int alpha, const cv::Scalar& color, int thickness, int lineType)
{
    Point arrow;
    double angle = atan2((double)(pStart.y - pEnd.y), (double)(pStart.x - pEnd.x));
    line(img, pStart, pEnd, color, thickness, lineType);
    arrow.x = pEnd.x + len * cos(angle + CV_PI * alpha / 180);
    arrow.y = pEnd.y + len * sin(angle + CV_PI * alpha / 180);
    line(img, pEnd, arrow, color, thickness, lineType);
    arrow.x = pEnd.x + len * cos(angle - CV_PI * alpha / 180);
    arrow.y = pEnd.y + len * sin(angle - CV_PI * alpha / 180);
    line(img, pEnd, arrow, color, thickness, lineType);
}

bool StereoLoc::judgeCorners(std::vector<cv::KeyPoint>& kpts, const cv::Mat& img)
{
    if(kpts.size() < 2)
        return false;
    if(kpts.size() == 2)
        return true;
    math_tools::GuassainDistribution theta_gd(0, 0.5);
    math_tools::LogisticRegression dist_lg(img.cols / 5, img.cols/ 20);
    float max_prob = -1;
    pair<int, int> max_prob_index(0, 1);
    for(int i = 0; i < kpts.size() - 1; i++)
        for(int j = i + 1; j < kpts.size(); j++)
        {
            float theta = atan((kpts[i].pt.y - kpts[j].pt.y) / (kpts[i].pt.x - kpts[j].pt.x));
            float theta_prob = theta_gd.CalcProbability(theta);
            float dist_prob = dist_lg.CalcProbability(fabs(kpts[i].pt.x - kpts[j].pt.x));
            float cur_prob = theta_prob * dist_prob;

            // cout << "theta_prob: " << theta_prob << endl;
            // cout << "dist_prob: " << dist_prob << endl;
            if(cur_prob > max_prob)
            {
                max_prob = cur_prob;
                max_prob_index.first = i;
                max_prob_index.second = j;
            }
        }
    kpts[0] = kpts[max_prob_index.first];
    kpts[1] = kpts[max_prob_index.second];
    kpts.resize(2);
    return true;
}

bool StereoLoc::judgeVerticalLines(vector<Vec2f>& vertical_lines, const Mat& img)
{

}

bool StereoLoc::judgeHorizonLines(vector<cv::Vec2f>& horizon_lines, const Mat& img)
{
    if(horizon_lines.size() < 1)
    {
        cout << "[judgeHorizonLines]: Can't find enough horizon_lines!" << endl;
        return false;
    }
    Mat horizon_edges;
    Sobel(img, horizon_edges, CV_8UC1, 0, 1, 3);
    // Canny(img, horizon_edges, Canny_lowThres_+40, Canny_highThres_ + 20);
    // Sobel(img, horizon_edges_16S, CV_16SC1, 0, 1, 3);
    
    if(!is_sim_)
    {
        threshold(horizon_edges, horizon_edges, 120, 255, THRESH_BINARY);
        Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
        Mat bin_img_closed;
        morphologyEx(horizon_edges, horizon_edges, MORPH_OPEN, element);
        erode(horizon_edges, horizon_edges, element, Point(-1, -1), 1);
        element = getStructuringElement(MORPH_RECT, Size(2, 2));
        erode(horizon_edges, horizon_edges, element, Point(-1, -1), 1);
    }

    // imshow("horizon_edges", horizon_edges_disp);
    // imshow("horizon_edges_16S", horizon_edges_16S);

    float max_prob = 0;
    int max_prob_index = 0;

    math_tools::LogisticRegression val_lr(128, 20), vote_lr(img.cols / 6, img.cols / 100);
    for(int i = 0; i < horizon_lines.size(); i++)
    {
        Mat line_mat(img.size(), CV_8UC1, Scalar(0));
        drawLine(horizon_lines[i], line_mat, 1, Scalar(255, 255, 255));
        Mat vote_mat;

        bitwise_and(line_mat, horizon_edges, vote_mat);
        Mat img_roi;
        img.copyTo(img_roi, vote_mat);
        Scalar edge_sum = sum(img_roi);

        int cur_vote = countNonZero(vote_mat);
        if(cur_vote == 0)
            continue;
        float average_val = edge_sum[0] / cur_vote;

        float vote_prob = vote_lr.CalcProbability(cur_vote);
        float val_prob = val_lr.CalcProbability(average_val);
        float cur_prob = vote_prob;


        /***** disp *****/
        // Mat horizon_edges_disp = horizon_edges.clone();
        // cvtColor(horizon_edges_disp, horizon_edges_disp, COLOR_GRAY2BGR);
        // drawLine(horizon_lines[i], horizon_edges_disp, 2, Scalar(255, 0, 0));
        // cout << cur_vote << endl;
        // resize(horizon_edges_disp, horizon_edges_disp, Size(), 0.5, 0.5);
        // imshow("horizon_edges", horizon_edges_disp);
        // waitKey(0);

        // cout << vote_prob << endl;
        // cout << val_prob << endl;
        // cout << cur_prob << endl;
        // cout << "-------" << endl;
        if(cur_prob > max_prob)
        {
            max_prob = cur_prob;
            max_prob_index = i;

        }
    }

    // cout << endl; 
    // cout << "max_prob_index: " << max_prob_index << endl;
    // cout << "max_prob: " << max_prob << endl;
    horizon_lines[0] = horizon_lines[max_prob_index];
    horizon_lines.resize(1);
    return true;
}

bool StereoLoc::hasInsection(cv::Vec2f& lhs, cv::Vec2f& rhs, int img_rows)
{
    float l_rho = lhs[0], l_theta = lhs[1];
    float l_x0 = cos(l_theta)*l_rho, l_y0 = sin(l_theta)*l_rho;
    float l_x_top = l_x0 + l_y0 * tan(l_theta);
    float l_x_bottom =  l_x0 + (l_y0 - img_rows) * tan(l_theta);

    float r_rho = rhs[0], r_theta = rhs[1];
    float r_x0 = cos(r_theta)*r_rho, r_y0 = sin(r_theta)*r_rho;
    float r_x_top = r_x0 + r_y0 * tan(r_theta);
    float r_x_bottom =  r_x0 + (r_y0 - img_rows) * tan(r_theta);

    // cout << l_x_top << ", " << r_x_top << endl;
    // cout << l_x_bottom << ", " << r_x_bottom << endl;
    // cout << "---------" << endl;

    if(fabs(l_x_top - r_x_top) <= 2)
        return false;

    if(fabs(l_x_bottom - r_x_bottom) <= 2)
        return false;

    if((l_x_top - r_x_top) * (l_x_bottom - r_x_bottom) >= 0)
        return false;
    else
        return true;
}

void StereoLoc::removeNet(const Mat& img, Mat& out_img)
{
    Mat gray_img;
    cvtColor(img, gray_img, COLOR_BGR2GRAY);
    // Mat gray_img_eq;
    // equalizeHist(gray_img, gray_img_eq);
    Mat thres_img;
    // threshold(gray_img, thres_img, 50, 255, THRESH_BINARY);
    adaptiveThreshold(gray_img, thres_img, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 31, -20);
    /***** disp_thres_img *****/
    // Mat thres_img_disp;
    // resize(thres_img, thres_img_disp, Size(), 0.5, 0.5);
    // imshow("thres_img", thres_img_disp);

    Mat element = getStructuringElement(MORPH_RECT, Size(5, 5));
    Mat thres_img_closed;
    morphologyEx(thres_img, thres_img_closed, MORPH_OPEN, element);
    Mat not_thres_img_closed;
    // bitwise_not(thres_img_closed, not_thres_img_closed);
    not_thres_img_closed = 255 - thres_img_closed;
    Mat net_mask;
    bitwise_and(not_thres_img_closed, thres_img, net_mask);

    // element = getStructuringElement(MORPH_RECT, Size(3, 3));
    // dilate(net_mask, net_mask, element);
    // resize(net_area, net_area, Size(), 0.5, 0.5);
    // Mat no_net_area = 255 - net_area;
    // Mat img_no_net;
    // img.copyTo(img_no_net, no_net_area);
    // resize(img_no_net, img_no_net, Size(), 0.5, 0.5);
    // imshow("img_no_net", img_no_net);
    // resize(thres_img, thres_img, Size(), 0.5, 0.5);
    // imshow("thres_img", thres_img);
    // // resize(thres_img_closed, thres_img_closed, Size(), 0.5, 0.5);
    // // imshow("thres_img_closed", thres_img_closed);

    out_img = Mat(img.size(), CV_8UC1, Scalar(0));
    gray_img.copyTo(out_img);
    for(int c = 0; c < out_img.cols; c++)
        for(int r = 0; r < out_img.rows; r++)
        {
            if(net_mask.at<uchar>(r, c) != 0)
            {
                int val_count = 0;
                float val = 0;
                for(int i = -5; i < 6; i++)
                    for(int j = -5; j < 6; j++)
                    {
                        if(i == 0 && j == 0)
                            continue;
                        int cur_r = r + i;
                        int cur_c = c + j;
                        if(cur_r < 0 || cur_r >= img.rows || cur_c < 0 || cur_c > img.cols)
                            continue;
                        if(net_mask.at<uchar>(cur_r, cur_c) != 0)
                            continue;
                        val += gray_img.at<uchar>(cur_r, cur_c);
                        val_count ++;
                        // val = gray_img.at<uchar>(cur_r, cur_c);

                    }
                out_img.at<uchar>(r, c) = val / val_count; 
            }
        }
    // Mat no_net_mask = 255 - net_mask;

    // Mat out_img_disp;
    // resize(out_img, out_img_disp, Size(), 0.5, 0.5);
    // imshow("no_net_img", out_img_disp);

    // resize(net_mask, net_mask, Size(), 0.5, 0.5);
    // imshow("net_mask", net_mask);
}


float StereoLoc::calcTwoVerticalLineDist(cv::Vec2f& lhs, cv::Vec2f& rhs, int img_rows)
{
    float l_rho = lhs[0], l_theta = lhs[1];
    float l_x0 = cos(l_theta)*l_rho, l_y0 = sin(l_theta)*l_rho;
    float l_x_top = l_x0 + l_y0 * tan(l_theta);
    float l_x_bottom =  l_x0 + (l_y0 - img_rows) * tan(l_theta);
    float l_dist = (l_x_top + l_x_bottom) / 2;

    float r_rho = rhs[0], r_theta = rhs[1];
    float r_x0 = cos(r_theta)*r_rho, r_y0 = sin(r_theta)*r_rho;
    float r_x_top = r_x0 + r_y0 * tan(r_theta);
    float r_x_bottom =  r_x0 + (r_y0 - img_rows) * tan(r_theta);
    float r_dist = (r_x_top + r_x_bottom) / 2;
    return fabs(l_dist - r_dist);
}


bool StereoLoc::checkEllipseShape(Mat src,vector<Point> contour,RotatedRect ellipse,double ratio)
{
	//get all the point on the ellipse point
	vector<Point> ellipse_point;

	//get the parameter of the ellipse
	Point2f center = ellipse.center;
	double a_2 = pow(ellipse.size.width*0.5,2);
	double b_2 = pow(ellipse.size.height*0.5,2);
	double ellipse_angle = (ellipse.angle*3.1415926)/180;
	
	//the uppart
	for(int i=0;i<ellipse.size.width;i++)
	{
		double x = -ellipse.size.width*0.5+i;
		double y_left = sqrt( (1 - (x*x/a_2))*b_2 );

        cv::Point2f rotate_point_left;
        rotate_point_left.x =  cos(ellipse_angle)*x - sin(ellipse_angle)*y_left;
        rotate_point_left.y = +sin(ellipse_angle)*x + cos(ellipse_angle)*y_left;

		//trans
		rotate_point_left += center;

		//store
		ellipse_point.push_back(Point(rotate_point_left));
	}
	//the downpart
	for(int i=0;i<ellipse.size.width;i++)
	{
		double x = ellipse.size.width*0.5-i;
		double y_right = -sqrt( (1 - (x*x/a_2))*b_2 );

        cv::Point2f rotate_point_right;
		rotate_point_right.x =  cos(ellipse_angle)*x - sin(ellipse_angle)*y_right;
        rotate_point_right.y = +sin(ellipse_angle)*x + cos(ellipse_angle)*y_right;

		//trans
		rotate_point_right += center;

		//store
		ellipse_point.push_back(Point(rotate_point_right));

	}


	vector<vector<Point> > contours1;
	contours1.push_back(ellipse_point);

	double a0 = matchShapes(ellipse_point,contour,CV_CONTOURS_MATCH_I1,0);  
	if (a0>0.01)
	{
		return true;      
	}

	return false;
}

int StereoLoc::fitLineRansac(const vector<Point>& points, Vec2f& res_line, int max_iter)
{
    int iter_num = min((ulong)max_iter, points.size() * (points.size()-1) / 2);
    int max_valid = 0;
    Vec4i best_line;
    RNG rng(19970730);
    for(int iter = 0; iter < max_iter; iter++)
    {
        int index1 = rng.uniform(0, points.size() - 1);
        int index2 = rng.uniform(0, points.size() - 1);
        if(index1 == index2)
        {
            iter -= 1;
            continue;
        }

        Vec4i cur_line(points[index1].x, points[index1].y, points[index2].x, points[index2].y);

        int valid_count = 0;
        for(int i = 0; i < points.size(); i++)
        {
            float a = (cur_line[3] - cur_line[1]);
            float b = -(cur_line[2] - cur_line[0]);;
            float c = cur_line[1]*cur_line[2] - cur_line[0]*cur_line[3];
            float dist = fabs(a * points[i].x + b * points[i].y + c) / sqrt(a*a + b*b);
            if(dist <= 1)
                valid_count ++;
        }
        if(valid_count > max_valid)
        {
            max_valid = valid_count;
            best_line = cur_line;
        }
    }

    float a = (best_line[3] - best_line[1]);
    float b = -(best_line[2] - best_line[0]);;
    float c = best_line[1]*best_line[2] - best_line[0]*best_line[3];
    res_line[0] = fabs(c) / sqrt(a*a + b*b);
    res_line[1] = atan((best_line[2] - best_line[0]) / (best_line[1] - best_line[3] + 0.000001));
    if(res_line[1] < -CV_PI / 4)
    {
        res_line[1] = CV_PI + res_line[1];
    }
    else if(res_line[1] > CV_PI / 4 * 3)
    {
        res_line[1] = CV_PI - res_line[1];
    }


    return max_valid;
    
}