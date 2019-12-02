/*
 * Created on Mon Dec 02 2019
 *
 * Copyright (c) 2019 HITSZ-NRSL
 * All rights reserved
 *
 * Author: EpsAvlc
 */

#include "goal_viewer.h"

#include <pangolin/pangolin.h>
#include <opencv2/core/core.hpp>
#include <thread>

using namespace std;
using namespace cv;

GoalViewer::GoalViewer(string& config_file_path)
{
    FileStorage fs(config_file_path, FileStorage::READ);
    assert(fs.isOpened());
    fs["window_width"] >> window_width_;
    fs["window_height"] >> window_height_;
    fs["viewer_point_x"] >> vp_x_;
    fs["viewer_point_y"] >> vp_y_;
    fs["viewer_point_z"] >> vp_z_;
    fs["viewer_point_f"] >> vp_f_;

    R_ = Eigen::Matrix3f::Identity();
    t_ = Eigen::Vector3f::Zero();
}

void GoalViewer::Run()
{
    pangolin::CreateWindowAndBind("StereoLoc: Goal Viewer", window_width_, window_height_);

    glEnable(GL_DEPTH_TEST);

    // Define Projection and initial ModelView matrix
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(window_width_, window_height_, vp_f_, vp_f_, window_width_ / 2 , window_height_ / 2, 0.2,100),
        pangolin::ModelViewLookAt(vp_x_, vp_y_, vp_z_, 0, 0, 0, 0.0, -1.0, 0.0)
    );

    // Create Interactive View in window
    pangolin::Handler3D handler(s_cam);
    pangolin::View& d_cam = pangolin::CreateDisplay().SetBounds(0.0, 1.0, 0.0, 1.0, -640.f/480.f).SetHandler(&handler);

    std::chrono::milliseconds dura(33);
    while( !pangolin::ShouldQuit() )
    {
        // Clear the window by current color.
        
        d_cam.Activate(s_cam);

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        drawGoal();
        pangolin::FinishFrame();
        // Mat cur_match_img = sfm_->GetCurMatch();
        // if(!cur_match_img.empty())
        // {
        //     lock_guard<mutex> lock(sfm_->viewer_mutex_);
        //     imshow("cur_match", cur_match_img);
        // }
        this_thread::sleep_for(dura);
    }
}

void GoalViewer::UpdatePose(const Eigen::Matrix3f& R, const Eigen::Vector3f& t)
{
    R_ = R;
    t_ = t;
}

void GoalViewer::drawGoal()
{
    const float goal_height = 2.0f;
    
    Eigen::Matrix4f cur_T;
    for(int i = 0; i < 3; i++)
        for(int j = 0; j < 3; j++)
        {
            cur_T(i, j) = R_(i, j);
        }
    cur_T(0, 3) = t_.x();
    cur_T(1, 3) = t_.y();
    cur_T(2, 3) = t_.z();
    cur_T(3, 3) = 1;

    pangolin::OpenGlMatrix cur_T_gl(cur_T);

    glPushMatrix();
    glMultMatrixd(cur_T_gl.m);

    glLineWidth(25);
    glColor3f(1, 1, 1);
    glBegin(GL_LINES);
    glVertex3f(0, 0, 0);
    glVertex3f(0, 0, -goal_height);
    glVertex3f(0, 0, 0);
    glVertex3f(0, 3, 0);
    glVertex3f(0, 3, 0);
    glVertex3f(0, 3, -goal_height);
    glEnd();

    glLineWidth(6);
    glBegin(GL_LINES);
    glVertex3f(0, 0, 0);
    glVertex3f(1.5, 0, 0);
    glVertex3f(1.5, 0, 0);
    glVertex3f(2, 0, -2);
    glVertex3f(2, 0, -2);
    glVertex3f(0, 0, -2);

    glVertex3f(0, 3, 0);
    glVertex3f(1.5, 3, 0);
    glVertex3f(1.5, 3, 0);
    glVertex3f(2, 3, -2);
    glVertex3f(2, 3, -2);
    glVertex3f(0, 3, -2);
    glVertex3f(0, 3, -2);
    glVertex3f(0, 0, -2);
    glEnd();
    glPopMatrix();
}