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
    blob_params_.filterByColor = true;
    blob_params_.blobColor = 0;

    blob_params_.filterByConvexity = true;
    blob_params_.minConvexity = 0.5;
    blob_params_.maxConvexity = 1;

    blob_params_.filterByInertia = true;
    blob_params_.minInertiaRatio = blob_minInertiaRatio_;
    blob_params_.maxInertiaRatio = 1;

    blob_params_.thresholdStep = 3;

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

    /***** line method ******/
    Eigen::MatrixXf left_P(3, 4); 
    left_P = Eigen::MatrixXf::Zero(3, 4);
    left_P.block(0, 0, 3, 3) = Eigen::Matrix3f::Identity();
    Eigen::Matrix3f left_K_eigen;
    cv2eigen(left_K_, left_K_eigen);
    left_P = left_K_eigen * left_P;
    bool left_successed = calcCornersByLine(left_img, left_corner_3d, right_corner_3d, left_P, left_corners_2d);

    // Eigen::MatrixXf right_P(3, 4);
    // right_P = Eigen::MatrixXf::Zero(3, 4);
    // right_P.block(0, 0, 3, 3) = R_;
    // right_P.block(0, 3, 3, 1) = t_;
    // Eigen::Matrix3f right_K_eigen;
    // cv2eigen(right_K_, right_K_eigen);
    // right_P = right_K_eigen * right_P;
    // bool right_successed = calcCornersByLine(right_img, left_corner_3d, right_corner_3d, right_P, right_corners_2d);

    // if(!(right_successed && left_successed))
    //     return false;

    // left_corner_3d = triangulation(left_corners_2d[0], right_corners_2d[0]);
    // right_corner_3d = triangulation(left_corners_2d[1], right_corners_2d[1]);

    // cout << left_corner_3d << endl;
    // cout << right_corner_3d << endl;

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

    // resize(left_img_clone, left_img_clone, Size(), 0.5, 0.5);
    imshow("goal", left_img_clone);
    goal_viewer_.UpdatePose(R, t);
}

bool StereoLoc::findCornerSubPix(const cv::Mat& img, vector<Point2f>& corners)
{
    vector<KeyPoint> key_corners;
    
    Mat thres_img;
    blob_detector_->detect(img, key_corners);

    for(auto it = key_corners.begin(); it != key_corners.end();)
    {
        uchar color = img.at<Vec3b>(it->pt)[0];
        if(color < keypoint_thres_)
        {
            it = key_corners.erase(it); 
        }
        else
        {
            ++it;
        }
    }

    /***** display keypoint *****/
    // Mat kp_image;
    // drawKeypoints(img, key_corners, kp_image, Scalar(255, 0, 0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    // resize(kp_image, kp_image, Size(), 0.5, 0.5);
    // imshow("keypoints", kp_image);
    // waitKey(0);

    // if(key_corners.size() != 2)
    // {
    //     cout << key_corners.size() << endl;
    //     return false;
    // }

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
    Mat bin_img;
    if(!is_sim_)
    {
        Mat thres_img;
        threshold(gray_img, thres_img, 200, 255, THRESH_BINARY);
        Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
        Mat bin_img_closed;
        morphologyEx(thres_img, thres_img, MORPH_OPEN, element);
        // imshow("Thres", thres_img);
        Canny(thres_img, bin_img, Canny_lowThres_, Canny_highThres_);
    }
    else
    {
        Canny(gray_img, bin_img, Canny_lowThres_, Canny_highThres_);
    }
    
    // Mat canny_img;
    // resize(bin_img, canny_img, Size(), 0.5, 0.5);
    imshow("canny", bin_img);

    /***** remove internal contours *****/
    // Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
    // Mat bin_img_closed;
    // morphologyEx(bin_img, bin_img_closed, MORPH_CLOSE, element);
    // vector<vector<Point>> contours;
    // vector<Vec4i> hierarchy;
    // findContours(bin_img_closed, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);
    // Mat contour_mat(img.size(), CV_8UC1, Scalar(0));
    // // for(int i = 0; i < contours.size(); i++)
    // // {
    // //     drawContours(contour_mat, contours, i, Scalar(255), 1, 8, hierarchy, 0);
    // // }
    // drawContours(contour_mat, contours, 0, Scalar(255));
    // drawContours(contour_mat, contours, hierarchy[0][3], Scalar(255));
    // imshow("contour_mat", contour_mat);
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
    line(line_mask, pts_2d[0], pts_2d[1], Scalar(255), line_roi_size_);
    line(line_mask, pts_2d[0], pts_2d[2], Scalar(255), line_roi_size_);
    line(line_mask, pts_2d[1], pts_2d[3], Scalar(255), line_roi_size_);
    
    Mat bin_img_roi;
    bin_img.copyTo(bin_img_roi, line_mask);

    /***** disp_roi ******/
    // Mat bin_img_roi_disp;
    // resize(bin_img_roi, bin_img_roi_disp, Size(), 0.5, 0.5);
    // imshow("bin_img_roi", bin_img_roi);

	vector<Vec2f> lines;
	HoughLines(bin_img_roi, lines, 1, CV_PI/180, Hough_minLength_, 20, 0);

    /***** Display hough lines *****/
    // drawLines(lines, gray_img, 1);

    /***** for vertical lines *****/
    vector<Vec2f> vertical_lines;
    for(int i = 0; i < lines.size(); i++)
    {
        float rho = lines[i][0], theta = lines[i][1];
        if((theta <= 0.03) || (CV_PI - theta <= 0.03))
            vertical_lines.push_back(lines[i]);
    }

    if(vertical_lines.size() < 4)
    {
        cout << "There are not enough vertical lines. " << endl; 
        return false;
    }
    // drawLines(vertical_lines, gray_img, 1);
    // cout << vertical_lines.size() << endl;

    sort(vertical_lines.begin(), vertical_lines.end(), [](Vec2f& lhs, Vec2f& rhs)
    {   
        return lhs[0] < rhs[0];
    });

    // int left_index = 0;
    // for(int i = 1; i < vertical_lines.size() - 1; i++)
    // {
    //     float left_dist = vertical_lines[i][0] - vertical_lines[0][0];
    //     float right_dist = vertical_lines[vertical_lines.size()-1][0] - vertical_lines[i][0];
    //     if(left_dist > right_dist || left_dist > 
    //     (vertical_lines[vertical_lines.size()-1][0] - vertical_lines[0][0])/9)
    //     {
    //         left_index = i-1;
    //         break;
    //     }
    // }

    // int right_index = 0;
    // for(int i = vertical_lines.size() - 2; i >= 1; i--)
    // {
    //     float left_dist = vertical_lines[i][0] - vertical_lines[0][0];
    //     float right_dist = vertical_lines[vertical_lines.size()-1][0] - vertical_lines[i][0];
    //     if(left_dist < right_dist || right_dist > 
    //     (vertical_lines[vertical_lines.size()-1][0] - vertical_lines[0][0])/9)
    //     {
    //         right_index = i+1;
    //         break;
    //     }
    // }

    // vertical_lines[1] = vertical_lines[left_index];
    // vertical_lines[2] = vertical_lines[right_index];
    // vertical_lines[3] = vertical_lines[vertical_lines.size()-1];
    // vertical_lines.resize(4);

    judgeVerticalLines(vertical_lines, img);

    drawLines(vertical_lines, gray_img, 1);

    /***** for horizon lines *****/
    vector<Vec2f> horizon_lines;
    for(int i = 0; i < lines.size(); i++)
    {
        float rho = lines[i][0], theta = lines[i][1];
        if(fabs(theta - CV_PI / 2) < 0.1 )
            horizon_lines.push_back(lines[i]);
    }

    judgeHorizonLines(horizon_lines, gray_img);
    drawLines(horizon_lines, gray_img, 1);

    // sort(horizon_lines.begin(), horizon_lines.end(), [](Vec2f& lhs, Vec2f& rhs)
    // {   
    //     return lhs[0] < rhs[0];
    // });
    // if(horizon_lines.size() > 2)
    // {
    //     swap(horizon_lines[1], horizon_lines[horizon_lines.size() - 1]);
    // }
    // if(horizon_lines.size() < 2)
    // {
    //         cout << "There are not enough horizon lines. " << endl; 
    //     return false;
    // }
    // horizon_lines.resize(1);
    
    Vec2f horizon_pillar_line = horizon_lines[0];
    
    Vec2f left_pillar_line, right_pillar_line;
    left_pillar_line = (vertical_lines[0] + vertical_lines[1]) / 2;
    right_pillar_line = (vertical_lines[2] + vertical_lines[3]) / 2;

    Point2f left_point = calcLineInsection(left_pillar_line, horizon_pillar_line);

    Point2f right_point = calcLineInsection(right_pillar_line, horizon_pillar_line);

    // resize(gray_img, gray_img, Size(), 0.5, 0.5);
    imshow("gray_img", gray_img);

    refine_corners_2d[0] = left_point;
    refine_corners_2d[1] = right_point;

    return true;
}

void StereoLoc::drawLines(const vector<Vec2f>& lines, Mat& out_img, int line_width)
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
		line(out_img, pt1, pt2, Scalar(255), line_width, CV_AA);
	}
}

void StereoLoc::drawLine(const Vec2f& l, Mat& out_img, int line_width)
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
    line(out_img, pt1, pt2, Scalar(255), line_width, CV_AA);
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

    float roll = 0.0;

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

bool StereoLoc::judgeVerticalLines(vector<Vec2f>& vertical_lines, const Mat& img)
{
    if(vertical_lines.size() < 4)
        return false;
    if(vertical_lines.size() == 4)
        return true;
    vector<pair<Vec2f, Vec2f>> line_pairs;
    for(int i = 0; i < vertical_lines.size() - 1; i++)
    {
        for(int j = i + 1; j < vertical_lines.size(); j++)
        {
            if(fabs(vertical_lines[i][0] - vertical_lines[j][0]) > img.rows / 3)
                continue;
            if(fabs(vertical_lines[i][0] - vertical_lines[j][0]) < 5)
                continue;
            if(hasInsection(vertical_lines[i], vertical_lines[j], img.rows))
                continue;
            line_pairs.push_back(make_pair(vertical_lines[i], vertical_lines[j]));
        }
    }

    math_tools::GuassainDistribution theta_gd(0, 0.05), rho_dist_gd(1700, 100), rho_diff_gd(0, 5);
    float max_prob = -1;
    pair<int, int> max_prob_index(-1, -1);
    for(int i = 0; i < line_pairs.size()-1; i++)
    {
        for(int j = i + 1; j < line_pairs.size(); j++)
        {
            // float theta_prob1 = theta_gd.CalcProbability(line_pairs[i].first[1] - line_pairs[i].second[1]);
            // float theta_prob2 = theta_gd.CalcProbability(line_pairs[j].first[1] - line_pairs[j].second[1]);
        
            float rho_diff_prob = rho_diff_gd.CalcProbability(fabs(line_pairs[i].first[0] - line_pairs[i].second[0]) - fabs(line_pairs[j].first[0] - line_pairs[j].second[0]));

            // float rho_dist_prob = rho_dist_gd.CalcProbability(line_pairs[i].first[0] - line_pairs[j].second[0]);

            if(fabs(line_pairs[i].first[0] - line_pairs[j].second[0]) < img.rows / 3)
                continue;

            float cur_prob = rho_diff_prob;

            // cout << "-------" << endl;
            // cout << line_pairs[i].first[1] - line_pairs[i].second[1] << ", " << theta_prob1 << endl;
            // cout << line_pairs[j].first[1] - line_pairs[j].second[1] << ", " << theta_prob2 << endl;
            // // cout << rho_diff_prob << endl;
            // cout << rho_dist_prob << endl;
            if(cur_prob > max_prob)
            {
                max_prob = cur_prob;
                max_prob_index.first = i;
                max_prob_index.second = j;
                // cout << rho_diff_prob << endl;
            }
        }
    }
    vertical_lines[0] = line_pairs[max_prob_index.first].first;
    vertical_lines[1] = line_pairs[max_prob_index.first].second;
    vertical_lines[2] = line_pairs[max_prob_index.second].first;
    vertical_lines[3] = line_pairs[max_prob_index.second].second;

    vertical_lines.resize(4);
}

bool StereoLoc::judgeHorizonLines(vector<cv::Vec2f>& horizon_lines, const Mat& img)
{
    Mat horizon_edges, horizon_edges_16S;
    Sobel(img, horizon_edges, CV_8UC1, 0, 1, 3);
    Sobel(img, horizon_edges_16S, CV_16SC1, 0, 1, 3);


    if(!is_sim_)
    {
        threshold(horizon_edges, horizon_edges, 120, 255, THRESH_BINARY);
        Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
        Mat bin_img_closed;
        morphologyEx(horizon_edges, horizon_edges, MORPH_OPEN, element);
        erode(horizon_edges, horizon_edges, element, Point(-1, -1), 1);
    }

    Mat horizon_edges_disp;
    // resize(horizon_edges, horizon_edges_disp, Size(), 0.5, 0.5);
    imshow("horizon_edges", horizon_edges);
    imshow("horizon_edges_16S", horizon_edges_16S);

    int max_vote = 0;
    int max_vote_index = 0;
    cout << horizon_lines.size() << endl;
    for(int i = 0; i < horizon_lines.size(); i++)
    {
        Mat line_mat(img.size(), CV_8UC1, Scalar(0));
        drawLine(horizon_lines[i], line_mat, 1);
        Mat vote_mat;
        //TODO: need to check.
        // bitwise_and(line_mat, horizon_edges, vote_mat);
        // Mat signed_vote_mat;
        // horizon_edges_16S.copyTo(signed_vote_mat, vote_mat);
        // Scalar edge_sum = sum(signed_vote_mat);
        // cout << "edge_sum: " << edge_sum << endl;
        // if(edge_sum[0] > 0)
        //     continue;
        // imshow("vote_mat", vote_mat);
        // waitKey(0);


        float rho = horizon_lines[i][0], theta = horizon_lines[i][1];
        // cout << theta << endl;
        double a = cos(theta), b = sin(theta);
        double x0 = a*rho, y0 = b*rho;
        cout << horizon_edges_16S.at<int>(y0, x0) << endl;

        int cur_vote = countNonZero(vote_mat);
        if(cur_vote > max_vote)
        {
            max_vote = cur_vote;
            max_vote_index = i;
        }
    }
    horizon_lines[0] = horizon_lines[max_vote_index];
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
    float r_x0 = cos(l_theta)*r_rho, r_y0 = sin(r_theta)*r_rho;
    float r_x_top = r_x0 + r_y0 * tan(r_theta);
    float r_x_bottom =  r_x0 + (r_y0 - img_rows) * tan(r_theta);

    // cout << l_x_top << ", " << r_x_top << endl;
    // cout << l_x_bottom << ", " << r_x_bottom << endl;
    // cout << "---------" << endl;

    if((l_x_top - r_x_top) * (l_x_bottom - r_x_bottom) >= 0)
        return false;
    else
        return true;
}