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
    /**
     * @brief Construct a new Stereo Loc object
     * 
     * @param config_file_path the path of the config file
     */
    StereoLoc(std::string config_file_path);
    /**
     * @brief Calculate the pose by stereo image
     * 
     * @param left_img [input] obtained by left camera
     * @param right_img [input] obtained by right camera
     * @return true Successfully calculate pose.
     * @return false Failed to calculate pose
     */
    bool CalcPose(const cv::Mat& left_img, const cv::Mat& right_img);
private:
    /**
     * @brief find the corners by a marker
     * 
     * @param img [input] input image
     * @param corners [output] output corners that detected from image.
     * @return true if sucessfully get corners.
     * @return false if failed to get corners.
     */
    bool findCornerSubPix(const cv::Mat& img, std::vector<cv::Point2f>& corners);
    /**
     * @brief Triangulate two correspond points.
     * 
     * @param l_p [input] left point
     * @param r_p [output] right point
     * @return cv::Point3f triangulated 3D point
     */
    cv::Point3f triangulation(const cv::Point2f& l_p, const cv::Point2f& r_p);
    /**
     * @brief Draw lines on image. The lines are in the form of rho/theta
     * 
     * @param lines [input] vector of Vec2f. 
     * @param out_img [input/output] image that where draw lines on.
     * @param line_width [input] line widthwe
     */
    void drawLines(const std::vector<cv::Vec2f>& lines, cv::Mat& out_img, int line_width);
    void drawLine(const cv::Vec2f& line, cv::Mat& out_img, int line_width);
    /**
     * @brief Use line insection to get corners.
     * 
     * @param img [input] input image
     * @param corners [output] output corners that detected from image. 
     * @return true if successfully get corners.
     * @return false if failed to get corners.
     */
    bool calcCornersByLine(const cv::Mat& img, const cv::Point3f& left_corner, const cv::Point3f& right_corner, const Eigen::MatrixXf& P, std::vector<cv::Point2f>& refine_corners_2d);
    /**
     * @brief calculate the intersection point of two lines.
     * 
     * @param line1 [input] line 1.
     * @param line2 [input] line 2.
     * @return cv::Point2f the intersection point.
     */
    cv::Point2f calcLineInsection(const cv::Vec2f& line1, const cv::Vec2f& line2);
    /**
     * @brief calculate the centre of gravity of a image.
     * 
     * @param img [input] input image.
     * @return cv::Point2f the centre of gravity.
     */
    cv::Point2f calcCentreOfGravity(const cv::Mat& img);
    /**
     * @brief Draw a goal on a image.
     * 
     * @param img [input/output] image
     * @param left_corner [input] left corner's 3d position
     * @param right_corner [input] right corner's 3d position
     * @param color [input] the color of the goal
     */
    void drawGoal(cv::Mat& img, const cv::Point3f& left_corner, const cv::Point3f& right_corner, const cv::Scalar& color);

    /**
     * @brief 
     * 
     * @param img 
     * @param pStart 
     * @param pEnd 
     * @param len 
     * @param alpha 
     * @param color 
     * @param thickness 
     * @param lintType 
     */
    void drawArrow(cv::Mat& img, cv::Point pStart, cv::Point pEnd, int len, int alpha, const cv::Scalar& color, int thickness = 1, int lintType = 8);

    bool judgeVerticalLines(std::vector<cv::Vec2f>& vertical_lines, const cv::Mat& img);

    bool judgeHorizonLines(std::vector<cv::Vec2f>& horizon_lines, const cv::Mat& img);

    bool hasInsection(cv::Vec2f& lhs, cv::Vec2f& rhs, int img_rows);

    cv::Ptr<cv::SimpleBlobDetector> blob_detector_;
    cv::SimpleBlobDetector::Params blob_params_;
    cv::Mat left_K_, right_K_;
    Eigen::Vector3f t_;
    Eigen::Matrix3f R_;
    GoalViewer goal_viewer_;
    std::thread viewer_thread_;
    float goal_height_, goal_length_, goal_width1_, goal_width2_, blob_minThres_, blob_maxThres_, blob_minArea_, blob_maxArea_, blob_minCircularity_, blob_minInertiaRatio_, Canny_lowThres_, Canny_highThres_, line_roi_size_, Hough_minLength_, keypoint_thres_;
    int is_sim_;
};