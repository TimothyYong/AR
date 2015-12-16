#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/line_descriptor/descriptor.hpp>
#include <opencv2/stitching/detail/motion_estimators.hpp>
#include <cvsba/cvsba.h>
#include <vector>
#include <string>
#include <iostream>
#include <sys/types.h>
#include <unistd.h>
#include "highgui.h"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
using namespace cv::flann;
using namespace cv::line_descriptor;
using namespace cv::detail;

Mat draw(Mat img, vector<Point2f> corners, vector<Point2f> imagePoints) {
//  for (int i = 0; i < imagePoints.size(); i++) {
//    circle(img, imagePoints[i], 4, Scalar(255, 0, 0));
//  }
  
  // draw lines
  for (int i = 0; i < 4; i++) {
    line(img, imagePoints[i], imagePoints[i+4], Scalar(255, 0, 0), 2);
  }
  for (int i = 0; i < 4; i++) {
    line(img, imagePoints[i], imagePoints[(i+1)%4], Scalar(0, 255, 0), 2);
  }
  for (int i = 0; i < 4; i++) {
    line(img, imagePoints[i+4], imagePoints[(i+1)%4+4], Scalar(0, 0, 255), 2);
  }
  return img;
}

int main(int argc, char *argv[]) {
  srand(getpid());

  arma::mat K = reshape(arma::mat({
        545.82463365389708, 0, 319.5,
        0, 545.82463365389708, 239.5,
        0, 0, 1 }), 3, 3).t();
  arma::mat D = arma::mat({ -0.17081096154528716, 0.26412699622915992,
      0, 0, -0.080381316677811496 }).t();
  Mat mtx = arma2opencv(K);
  Mat dist = arma2opencv(D);

  vector<string> names = { "camcali/img10.png", "camcali/img15.png" };

  int w = 8;
  int h = 6;
  // termination criteria
  TermCriteria criteria(3, 30, 0.1);
  // prepare object points
  vector<Point3f> objp;
  for (int i = 0; i < w * h; i++) {
    objp.push_back(Point3f((float)(i % w), (float)(i / w), 0));
  }

  for (string fname : names) {
    Mat img = imread(fname);
    Mat gray;
    cvtColor(img, gray, CV_BGR2GRAY);

    vector<Point2f> corners;
    bool ret = findChessboardCorners(gray, Size(w, h), corners);

    if (ret) {
      // get the corners
      cornerSubPix(gray, corners, Size(11,11), Size(-1,-1), criteria);

      Mat rvec = Mat::zeros(3, 1, CV_64F);
      Mat tvec = Mat::zeros(3, 1, CV_64F);

      // find the R and t
      solvePnPRansac(objp, corners, mtx, dist, rvec, tvec);

      // list cube
      vector<Point3f> axis;
      axis.push_back(Point3f(0, 0, 0));
      axis.push_back(Point3f(0, 3, 0));
      axis.push_back(Point3f(3, 3, 0));
      axis.push_back(Point3f(3, 0, 0));
      axis.push_back(Point3f(0, 0, -3));
      axis.push_back(Point3f(0, 3, -3));
      axis.push_back(Point3f(3, 3, -3));
      axis.push_back(Point3f(3, 0, -3));
      vector<Point2f> imagePoints;

      // project 3d points to image plane and draw it
      projectPoints(axis, rvec, tvec, mtx, dist, imagePoints);
      Mat newimg = draw(img, corners, imagePoints);
      imwrite("before_cube.png", newimg);

      // create homography for image
      Mat image = imread("one-punch.jpg");
      vector<Point2f> src = { Point2f(0, 0), Point2f(0, image.rows), Point2f(image.cols, 0), Point2f(image.cols, image.rows) };
      vector<Point2f> dst = { imagePoints[4], imagePoints[5], imagePoints[7], imagePoints[6] };
      Mat h = findHomography(src, dst);
      Mat subimage;
      warpPerspective(image, subimage, h, newimg.size());
      Mat mask = Mat::ones(image.rows, image.cols, CV_8UC1);
      Mat submask;
      warpPerspective(mask, submask, h, newimg.size());
      subimage.copyTo(newimg, submask);
      imwrite("after_cube.png", newimg);

      imshow("image", newimg);
      waitKey(0);
    }

    destroyAllWindows();

  }

  return 0;
}
