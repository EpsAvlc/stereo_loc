/*
 * Created on Mon Dec 02 2019
 *
 * Copyright (c) 2019 HITSZ-NRSL
 * All rights reserved
 *
 * Author: EpsAvlc
 */

#include <iostream>
#include <Eigen/Core>

class GoalViewer
{
public:
    GoalViewer(std::string& config_file_path);
    void Run();
    void UpdatePose(const Eigen::Matrix3f& R, const Eigen::Vector3f& t);
private:
    void drawGoal();
    int window_width_, window_height_;
    float vp_x_, vp_y_, vp_z_, vp_f_;
    Eigen::Matrix3f R_;
    Eigen::Vector3f t_;
};
