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
    fs["baseline"] >> baseline_;
    fs["goal_height"] >> goal_height_;
    fs["goal_width1"] >> goal_width1_;
    fs["goal_width2"] >> goal_width2_;
    fs["goal_length"] >> goal_length_;

    /***** Init blob parameters *****/
    blob_params_.minThreshold = 10;
    blob_params_.maxThreshold = 100;
    blob_params_.filterByArea = true;
    blob_params_.minArea = 100;
    blob_params_.maxArea = 1000000;
    blob_params_.filterByCircularity = true;
    blob_params_.minCircularity = 0.7;
    blob_params_.filterByColor = true;
    blob_params_.blobColor = 0;

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

    Eigen::MatrixXf left_P(3, 4); 
    left_P = Eigen::MatrixXf::Zero(3, 4);
    left_P.block(0, 0, 3, 3) = Eigen::Matrix3f::Identity();
    Eigen::Matrix3f left_K_eigen;
    cv2eigen(left_K_, left_K_eigen);
    left_P = left_K_eigen * left_P;
    bool left_successed = calcCornersByLine(left_img, left_corner_3d, right_corner_3d, left_P, left_corners_2d);

    Eigen::MatrixXf right_P(3, 4);
    right_P = Eigen::MatrixXf::Zero(3, 4);
    right_P.block(0, 0, 3, 3) = Eigen::Matrix3f::Identity();
    right_P(0, 3) = baseline_;
    Eigen::Matrix3f right_K_eigen;
    cv2eigen(right_K_, right_K_eigen);
    right_P = right_K_eigen * right_P;
    bool right_successed = calcCornersByLine(right_img, left_corner_3d, right_corner_3d, right_P, right_corners_2d);

    if(!(right_successed && left_successed))
        return false;

    left_corner_3d = triangulation(left_corners_2d[0], right_corners_2d[0]);
    right_corner_3d = triangulation(left_corners_2d[1], right_corners_2d[1]);

    /***** for viewer *****/
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

    goal_viewer_.UpdatePose(R, t);
}

bool StereoLoc::findCornerSubPix(const cv::Mat& img, vector<Point2f>& corners)
{
    vector<KeyPoint> key_corners;
    
    blob_detector_->detect(img, key_corners);

    /***** display keypoint *****/
    Mat kp_image;
    drawKeypoints(img, key_corners, kp_image, Scalar(255, 0, 0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    // imshow("keypoints", kp_image);
    // waitKey(0);

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
        // GaussianBlur(roi, roi, Size(3,3), 1);
        // imshow("roi", roi);
        Mat bin_roi;
        threshold(roi, bin_roi, 4, 255, THRESH_BINARY_INV|THRESH_OTSU);
        // imshow("bin_roi", bin_roi);

        Point2f final_corner = calcCentreOfGravity(bin_roi);

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

    Eigen::Matrix3f R = Eigen::Matrix3f::Identity();
    Eigen::Vector3f t = Eigen::Vector3f::Zero();
    t.x() = baseline_;

    Eigen::Vector3f A = r_x_hat * R * l_x_3;
    Eigen::Vector3f b = -r_x_hat * t; 
    float s2 = ((A.transpose() * A).inverse() * (A.transpose() * b))(0, 0);
    A = R * l_x_3;
    b = s2*r_x_3 - t;
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
    Mat bin_img;
    // adaptiveThreshold(gray_img, bin_img, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 3, 0);
    Canny(gray_img, bin_img, 25, 150);
    imshow("canny", bin_img);

    /***** remove internal contours *****/
    Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
    Mat bin_img_closed;
    morphologyEx(bin_img, bin_img_closed, MORPH_CLOSE, element);
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(bin_img_closed, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
    Mat contour_mat(img.size(), CV_8UC1, Scalar(0));
    // for(int i = 0; i < contours.size(); i++)
    // {
    //     drawContours(contour_mat, contours, i, Scalar(255), 1, 8, hierarchy, 0);
    // }
    drawContours(contour_mat, contours, 0, Scalar(255));
    drawContours(contour_mat, contours, hierarchy[0][3], Scalar(255));
    imshow("contour_mat", contour_mat);
    /***** construct mask *****/
    Eigen::MatrixXf pts_4d(4, 4);
    pts_4d(0, 0) = left_corner_3d.x;
    pts_4d(1, 0) = left_corner_3d.y;
    pts_4d(2, 0) = left_corner_3d.z;
    pts_4d(3, 0) = 1;
    pts_4d(0, 1) = right_corner_3d.x;
    pts_4d(1, 1) = right_corner_3d.y;
    pts_4d(2, 1) = right_corner_3d.z;
    pts_4d(3, 1) = 1;
    pts_4d(0, 2) = left_corner_3d.x;
    pts_4d(1, 2) = left_corner_3d.y + goal_height_;
    pts_4d(2, 2) = left_corner_3d.z;
    pts_4d(3, 2) = 1;
    pts_4d(0, 3) = pts_4d(0, 1);
    pts_4d(1, 3) = pts_4d(1, 1) + goal_height_;
    pts_4d(2, 3) = pts_4d(2, 1);
    pts_4d(3, 3) = 1;

    Eigen::MatrixXf pts_homo_2d = P * pts_4d;
    vector<Point2f> pts_2d;
    for(int i = 0; i < pts_4d.cols(); i++)
    {
        Point2f cur_pt;
        cur_pt.x = pts_homo_2d(0, i) / pts_homo_2d(2, i); 
        cur_pt.y = pts_homo_2d(1, i) / pts_homo_2d(2, i); 
        pts_2d.push_back(cur_pt);
    }

    Mat line_mask(img.size(), CV_8UC1, Scalar(0));
    line(line_mask, pts_2d[0], pts_2d[1], Scalar(255), 50);
    line(line_mask, pts_2d[0], pts_2d[2], Scalar(255), 50);
    line(line_mask, pts_2d[1], pts_2d[3], Scalar(255), 50);
    
    Mat bin_img_roi;
    bin_img.copyTo(bin_img_roi, line_mask);
    // cout << bin_img_roi.channels() << endl;

	vector<Vec2f> lines;
	HoughLines(bin_img_roi, lines, 1, CV_PI/180, 100, 20, 0);
    /***** for vertical lines *****/
    vector<Vec2f> vertical_lines;
    for(int i = 0; i < lines.size(); i++)
    {
        float rho = lines[i][0], theta = lines[i][1];
        if((theta <= 0.01) || (CV_PI - theta <= 0.01))
            vertical_lines.push_back(lines[i]);
    }

    if(vertical_lines.size() < 4)
    {
        return false;
    }

    sort(vertical_lines.begin(), vertical_lines.end(), [](Vec2f& lhs, Vec2f& rhs)
    {   
        return lhs[0] < rhs[0];
    });

    int left_index = 0;
    for(int i = 1; i < vertical_lines.size() - 1; i++)
    {
        float left_dist = vertical_lines[i][0] - vertical_lines[0][0];
        float right_dist = vertical_lines[vertical_lines.size()-1][0] - vertical_lines[i][0];
        if(left_dist > right_dist || left_dist > 
        (vertical_lines[vertical_lines.size()-1][0] - vertical_lines[0][0])/9)
        {
            left_index = i-1;
            break;
        }
    }

    int right_index = 0;
    for(int i = vertical_lines.size() - 2; i >= 1; i--)
    {
        float left_dist = vertical_lines[i][0] - vertical_lines[0][0];
        float right_dist = vertical_lines[vertical_lines.size()-1][0] - vertical_lines[i][0];
        if(left_dist < right_dist || right_dist > 
        (vertical_lines[vertical_lines.size()-1][0] - vertical_lines[0][0])/9)
        {
            right_index = i+1;
            break;
        }
    }
    vertical_lines[1] = vertical_lines[left_index];
    vertical_lines[2] = vertical_lines[right_index];
    vertical_lines[3] = vertical_lines[vertical_lines.size()-1];
    vertical_lines.resize(4);

    drawLines(vertical_lines, gray_img, 1);

    /***** for horizon lines *****/
    vector<Vec2f> horizon_lines;
    for(int i = 0; i < lines.size(); i++)
    {
        float rho = lines[i][0], theta = lines[i][1];
        if(fabs(theta - CV_PI / 2) < 0.2 )
            horizon_lines.push_back(lines[i]);
    }

    sort(horizon_lines.begin(), horizon_lines.end(), [](Vec2f& lhs, Vec2f& rhs)
    {   
        return lhs[0] < rhs[0];
    });
    if(horizon_lines.size() > 2)
    {
        swap(horizon_lines[1], horizon_lines[horizon_lines.size() - 1]);
    }
    if(horizon_lines.size() < 2)
    {
        return false;
    }
    horizon_lines.resize(2);
    
    Vec2f horizon_pillar_line;
    horizon_pillar_line = (horizon_lines[0] + horizon_lines[1]) / 2;
    
    Vec2f left_pillar_line, right_pillar_line;
    left_pillar_line = (vertical_lines[0] + vertical_lines[1]) / 2;
    right_pillar_line = (vertical_lines[2] + vertical_lines[3]) / 2;

    Point2f left_point = calcLineInsection(left_pillar_line, horizon_pillar_line);

    Point2f right_point = calcLineInsection(right_pillar_line, horizon_pillar_line);

    drawLines(horizon_lines, gray_img, 1);

    imshow("gray_img", gray_img);

    refine_corners_2d[0] = left_point;
    refine_corners_2d[1] = right_point;
}

void StereoLoc::drawLines(const vector<Vec2f> lines, Mat& out_img, int line_width)
{
    for( size_t i = 0; i < lines.size(); i++ )
	{
		float rho = lines[i][0], theta = lines[i][1];
        // cout << theta << endl;
		Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a*rho, y0 = b*rho;
		pt1.x = cvRound(x0 + 1000*(-b));
		pt1.y = cvRound(y0 + 1000*(a));
		pt2.x = cvRound(x0 - 1000*(-b));
		pt2.y = cvRound(y0 - 1000*(a));
		line(out_img, pt1, pt2, Scalar(255), line_width, CV_AA);
	}
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
    pts_3d(1, 4) = pts_3d(1, 0);
    pts_3d(2, 4) = pts_3d(2, 0) + goal_width2_ * cos(pitch);    
    pts_3d(0, 5) = pts_3d(0, 4) + goal_length_ * cos(pitch);
    pts_3d(1, 5) = pts_3d(1, 4);
    pts_3d(2, 5) = pts_3d(2, 4) + goal_length_ * sin(pitch);
    pts_3d(0, 6) = pts_3d(0, 2) - goal_width1_ * sin(pitch);
    pts_3d(1, 6) = pts_3d(1, 2);
    pts_3d(2, 6) = pts_3d(2, 2) + goal_width1_ * cos(pitch);
    pts_3d(0, 7) = pts_3d(0, 3) - goal_width1_ * sin(pitch);
    pts_3d(1, 7) = pts_3d(1, 3);
    pts_3d(2, 7) = pts_3d(2, 3) + goal_width1_ * cos(pitch);
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
    line(img, pts_2d[5], pts_2d[4], color, 3);
    line(img, pts_2d[6], pts_2d[4], color, 3);
    line(img, pts_2d[7], pts_2d[5], color, 3);
    line(img, pts_2d[1], pts_2d[5], color, 3);
    line(img, pts_2d[2], pts_2d[6], color, 3);
    line(img, pts_2d[3], pts_2d[7], color, 3);
    line(img, pts_2d[6], pts_2d[7], color, 3);

    drawArrow(img, pts_2d[8], pts_2d[9], 17, 15, Scalar(255, 0, 0), 2);
    drawArrow(img, pts_2d[8], pts_2d[10], 17, 15, Scalar(0, 255, 0), 2);
    drawArrow(img, pts_2d[8], pts_2d[11], 17, 15, Scalar(255, 0, 255), 2);

    imshow("line", img);
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