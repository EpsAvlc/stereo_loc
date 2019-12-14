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

Vec2f FitLineRansac(const vector<Point>& points, int max_iter = 1000)
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

    Vec2f res;
    float a = (best_line[3] - best_line[1]);
    float b = -(best_line[2] - best_line[0]);;
    float c = best_line[1]*best_line[2] - best_line[0]*best_line[3];
    res[0] = fabs(c) / sqrt(a*a + b*b);
    res[1] = atan(b / (a + 0.000001));

    if(res[1] < -CV_PI / 4)
    {
        res[1] = CV_PI + res[1];
    }
    else if(res[1] > CV_PI / 4 * 3)
    {
        res[1] = CV_PI - res[1];
    }
    return res;
    
}


void DrawLine(const Vec2f& l, Mat& out_img, int line_width, const Scalar& color)
{
    float rho = l[0], theta = l[1];
    // cout << theta << endl;
 
    double a = cos(theta), b = sin(theta);
    float x0 = a * rho, y0 = b * rho;
    
    float k = -a/(b + 0.0000001);
    Point pt1, pt2;
    if(fabs(k) <= (out_img.rows / (float)out_img.cols))
    {
        pt1.x = 0;
        pt1.y = y0 + ((x0 - pt1.x) / b) * a;
        pt2.x = out_img.cols;
        pt2.y = y0 + ((x0 - pt2.x) / b) * a;
    }
    else
    {
        pt1.y = 0;
        pt1.x = x0 - ((pt1.y - y0) / a) * b;
        pt2.y = out_img.rows;
        pt2.x = x0 - ((pt2.y - y0) / a) * b;
    }
    
    line(out_img, pt1, pt2, color, line_width, CV_AA);
}


int main()
{
    Mat src = imread("/home/cm/Workspaces/stereo_loc/src/stereo_loc/imgs/line_roi.png");
    imshow("src", src);

    // Mat roi = src(Rect(0, 400, 300, 300));
    Mat roi = src.clone();
    imshow("roi", roi);
    cvtColor(roi, roi, COLOR_BGR2GRAY);
    vector<vector<Point>> contours;
    findContours(roi, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
    cvtColor(roi, roi, COLOR_GRAY2BGR);

    for(int i = 0; i < contours.size(); i++)
    {
        if(contours[i].size() < 100)
            continue;
        Vec2f lineP = FitLineRansac(contours[i]);
        cout << lineP << endl;
        Mat contour_mat(roi.size(), roi.type(), Scalar(0, 0, 0));
        drawContours(contour_mat, contours, i, Scalar(255, 255,255));
        
        DrawLine(lineP, contour_mat, 1, Scalar(0, 0, 255));
        imshow("lines", contour_mat);
        waitKey(0);
    }

}