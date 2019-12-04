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
    blob_params_.maxThreshold = 100;
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

    cout << left_corner_3d << endl;
    cout << right_corner_3d << endl;

    Eigen::Vector3f t;
    t.x() = left_corner_3d.x;
    t.y() = left_corner_3d.y;
    t.z() = left_corner_3d.z;
    
    float yaw = atan((left_corner_3d.z - right_corner_3d.z) 
    / fabs(left_corner_3d.x - right_corner_3d.x));

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

    calcCornersByLine(left_img, left_corners_2d);
    calcCornersByLine(right_img, right_corners_2d);

    left_corner_3d = triangulation(left_corners_2d[0], right_corners_2d[0]);
    right_corner_3d = triangulation(left_corners_2d[1], right_corners_2d[1]);

    cout << left_corner_3d << endl;
    cout << right_corner_3d << endl;

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
    float s1 = 0;
    s1 = ((A.transpose() * A).inverse() * (A.transpose() * b))(0, 0);
    return Point3f(s1*r_x_3.x(), s1*r_x_3.y(), s1*r_x_3.z());
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

bool StereoLoc::calcCornersByLine(const cv::Mat& img,  vector<Point2f>& corners)
{
    Mat gray_img;
    cvtColor(img, gray_img, COLOR_BGR2GRAY);
    Mat bin_img;
    // adaptiveThreshold(gray_img, bin_img, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 3, 0);
    Canny(gray_img, bin_img, 50, 150);
    imshow("canny", bin_img);

	vector<Vec2f> lines;
	HoughLines(bin_img, lines, 1, CV_PI/180, 125, 0, 0);

    /***** for vertical lines *****/
    vector<Vec2f> vertical_lines;
    for(int i = 0; i < lines.size(); i++)
    {
        float rho = lines[i][0], theta = lines[i][1];
        if((theta <= 0.01) || (CV_PI - theta <= 0.01))
            vertical_lines.push_back(lines[i]);
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
    // cout << middle_index << endl;
    vertical_lines[1] = vertical_lines[left_index];
    vertical_lines[2] = vertical_lines[right_index];
    vertical_lines[3] = vertical_lines[vertical_lines.size()-1];
    vertical_lines.resize(4);

    // drawLines(vertical_lines, gray_img, 1);

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
    horizon_lines.resize(2);
    
    Vec2f horizon_pillar_line;
    horizon_pillar_line = (horizon_lines[0] + horizon_lines[1]) / 2;
    
    Vec2f left_pillar_line, right_pillar_line;
    left_pillar_line = (vertical_lines[0] + vertical_lines[1]) / 2;
    right_pillar_line = (vertical_lines[2] + vertical_lines[3]) / 2;

    Point2f left_point = calcLineInsection(left_pillar_line, horizon_pillar_line);

    Point2f right_point = calcLineInsection(right_pillar_line, horizon_pillar_line);

    circle(gray_img, left_point, 3, Scalar(255), -1);
    circle(gray_img, right_point, 3, Scalar(255), -1);

    imshow("gray_img", gray_img);

    corners[0] = left_point;
    corners[1] = right_point;
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