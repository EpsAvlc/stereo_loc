/*
 * Created on Sat Nov 30 2019
 *
 * Copyright (c) 2019 HITSZ-NRSL
 * All rights reserved
 *
 * Author: EpsAvlc
 */


#include <ros/ros.h>
#include <iostream>
#include <sensor_msgs/Image.h>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include "stereo_loc.h"

using namespace std;

class Location
{
public:
    Location(ros::NodeHandle& nh, ros::NodeHandle& nh_local);
private:
    void readParam();
    ros::NodeHandle nh_, nh_local_;
    message_filters::Subscriber<sensor_msgs::Image> left_img_sub_, right_img_sub_;
    string left_cam_topic_, right_cam_topic_;

    // ApoproximateTime policy
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> stereo_policy; 
    typedef message_filters::Synchronizer<stereo_policy> stereo_sync;
    shared_ptr<stereo_sync> sync_;
    void stereoCB(const sensor_msgs::ImageConstPtr& left_image, const sensor_msgs::ImageConstPtr& right_image);

    void leftCB(const sensor_msgs::ImageConstPtr& left_image);

};

Location::Location(ros::NodeHandle& nh, ros::NodeHandle& nh_local) : nh_(nh), nh_local_(nh_local)
{
    readParam();
    left_img_sub_.subscribe(nh_, left_cam_topic_ + "/image_raw", 5);
    right_img_sub_.subscribe(nh_, left_cam_topic_ + "/image_raw", 5);

    // cout << left_cam_topic_ + "/image_raw" << endl;    
    sync_.reset(new stereo_sync(stereo_policy(10), left_img_sub_, right_img_sub_));
    sync_->registerCallback(boost::bind(&Location::stereoCB, this, _1, _2));
    cout << "register fishend" << endl;
}

void Location::readParam()
{
    nh_local_.param<string>("left_cam_topic", left_cam_topic_, "left_camera");
    nh_local_.param<string>("right_cam_topic", right_cam_topic_, "right_camera");
}

void Location::stereoCB(const sensor_msgs::ImageConstPtr& left_image, const sensor_msgs::ImageConstPtr& right_image)
{
    
}

int main(int argc, char**argv)
{
    ros::init(argc, argv, "location_node");
    ros::NodeHandle nh, nh_local("");
    Location loc(nh, nh_local);
    ros::spin();
}