/*
 * Created on Thu Dec 12 2019
 *
 * Copyright (c) 2019 HITSZ-NRSL
 * All rights reserved
 *
 * Author: EpsAvlc
 */

#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

Vec4i FitLineRansac(const vector<Point>& points, int max_iter = 1000)
{
    int iter_num = min((ulong)max_iter, points.size() * (points.size()-1) / 2);
    int max_valid = 0;
    Vec4i best_line;
    RNG rng(19053096);
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

    return best_line;
    
}

int main()
{
    Mat src = imread("/home/cm/Workspaces/stereo_loc/src/stereo_loc/imgs/line_roi.png");
    imshow("src", src);

    Mat roi = src(Rect(0, 527, 1000, 300));
    imshow("roi", roi);
    cvtColor(roi, roi, COLOR_BGR2GRAY);
    vector<vector<Point>> contours;
    findContours(roi, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    cvtColor(roi, roi, COLOR_GRAY2BGR);

    for(int i = 0; i < contours.size(); i++)
    {
        if(contours[i].size() < 20)
            continue;
        Vec4i lineP = FitLineRansac(contours[i]);

        Mat contour_mat(roi.size(), roi.type(), Scalar(0, 0, 0));
        drawContours(contour_mat, contours, i, Scalar(255, 255,255));
        line(contour_mat, Point(lineP[0], lineP[1]), Point(lineP[2], lineP[3]), Scalar(0, 0, 255), 2);
        imshow("lines", contour_mat);
        waitKey(0);
    }

}