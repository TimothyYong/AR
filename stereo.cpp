#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/line_descriptor/descriptor.hpp>
#include <opencv2/stitching/detail/motion_estimators.hpp>
#include <cvsba/cvsba.h> // used for the clustering
#include <vector>
#include <string>
#include <iostream>
#include <sys/types.h>
#include <unistd.h>
#include "highgui.h"
#include "kcluster.h"

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
using namespace cv::flann;
using namespace cv::line_descriptor;
using namespace cv::detail;
const int nmatches = 35;

// Camera Matrix for the PS3 Eye
Mat mtx(3, 3, CV_64F);

// Distortion Coefficients for the PS3 Eye
Mat dist(5, 1, CV_64F);

Mat drawMatches(Mat img1, vector<KeyPoint> kp1, Mat img2, vector<KeyPoint> kp2, vector<DMatch> matches) {
  int rows1 = img1.rows;
  int cols1 = img1.cols;
  int rows2 = img2.rows;
  int cols2 = img2.cols;

  Mat color1, color2;
  cvtColor(img1, color1, COLOR_GRAY2BGR);
  cvtColor(img2, color2, COLOR_GRAY2BGR);
  Mat out(MAX(rows1, rows2), cols1+cols2, CV_8UC3);
  for (int i = 0; i < rows1; i++) {
    for (int j = 0; j < cols1; j++) {
      out.at<Vec3b>(i, j) = color1.at<Vec3b>(i, j);
    }
  }
  for (int i = 0; i < rows2; i++) {
    for (int j = 0; j < cols2; j++) {
      out.at<Vec3b>(i, j+cols1) = color2.at<Vec3b>(i, j);
    }
  }

  for (DMatch m : matches) {
    int img1_idx = m.queryIdx;
    int img2_idx = m.trainIdx;

    int x1 = (int)kp1[img1_idx].pt.x;
    int y1 = (int)kp1[img1_idx].pt.y;
    int x2 = (int)kp2[img2_idx].pt.x;
    int y2 = (int)kp2[img2_idx].pt.y;

    Vec3b color(rand() % 255, rand() % 255, rand() % 255);
    circle(out, Point(x1,y1), 4, color, 1);
    circle(out, Point(x2+cols1,y1), 4, color, 1);
    line(out, Point(x1,y1), Point(x2+cols1,y2), color, 1);
  }
  return out;
}

void drawLines(Mat img1, Mat img2, Mat lines, vector<Point2f> pts1, vector<Point2f> pts2, Mat &ret1, Mat &ret2) {
  int n_rows = img1.rows;
  int n_cols = img1.cols;
  Mat color1, color2;
  cvtColor(img1, color1, COLOR_GRAY2BGR);
  cvtColor(img2, color2, COLOR_GRAY2BGR);
  for (int i = 0; i < lines.rows; i++) {
    Vec3f r = lines.at<Vec3f>(i, 0);
    Point2f pt1 = pts1[i];
    Point2f pt2 = pts2[i];
    Vec3b color(rand() % 255, rand() % 255, rand() % 255);
    int x0 = 0;
    int y0 = -r[2] / r[1];
    int x1 = n_cols;
    int y1 = -(r[2]+r[0]*n_cols)/r[1];
    line(color1, Point(x0,y0), Point(x1,y1), color, 1);
    circle(color1, pt1, 5, color, -1);
    circle(color2, pt2, 5, color, -1);
  }
  ret1 = color1;
  ret2 = color2;
}

arma::mat compute_depth(arma::mat disparity) {
  arma::mat depth(disparity.n_rows, disparity.n_cols, arma::fill::zeros);
  disparity /= disparity.max(); // normalize
  for (arma::uword i = 0; i < disparity.n_rows; i++) {
    for (arma::uword j = 0; j < disparity.n_cols; j++) {
      if (disparity(i, j) == 0.0) {
        depth(i, j) = 0.0;
      } else {
        depth(i, j) = 1.0 / disparity(i, j);
      }
    }
  }
  depth /= depth.max();
  return depth;
}

Mat depth2(Mat leftImage, Mat rightImage, Mat &R, Mat &t, Mat &pts3d, vector<Point2f> pts1, vector<Point2f> pts2) {
  // focal and point projective
  Point2d pp(mtx.at<double>(0, 2), mtx.at<double>(1, 2));
  double focal = mtx.at<double>(0, 0);

  // Start the SIFT detector
  Ptr<SIFT> sift = SIFT::create();

  // Find the keypoints from SIFT using the images given
  vector<KeyPoint> kp1, kp2;
  Mat des1, des2;
  sift->detectAndCompute(leftImage, noArray(), kp1, des1);
  sift->detectAndCompute(rightImage, noArray(), kp2, des2);

  // FLANN parameters
  Ptr<IndexParams> indexParams = makePtr<KDTreeIndexParams>(5);
  Ptr<SearchParams> searchParams = makePtr<SearchParams>(50);
  FlannBasedMatcher flann(indexParams, searchParams);
  vector< vector<DMatch> > matches;
  flann.knnMatch(des1, des2, matches, 2);

  vector<DMatch> good;

  // ratio test (as per Lowe's paper)
  for (int i = 0; i < matches.size(); i++) {
    DMatch m = matches[i][0];
    DMatch n = matches[i][1];
    if (m.distance < 0.8 * n.distance) {
      good.push_back(m);
      pts2.push_back(kp2[m.trainIdx].pt);
      pts1.push_back(kp1[m.queryIdx].pt);
    }
  }

  Mat matchFrame = drawMatches(leftImage, kp1, rightImage, kp2, good);
  imwrite("matches.png", matchFrame);

  // find the fundamental matrix, then use H&Z for essential
  //Mat E = findEssentialMat(pts1, pts2, focal, pp);
  Mat F = findFundamentalMat(pts1, pts2, FM_RANSAC, 3.0, 0.99);
  Mat E = mtx.t() * F * mtx;
  arma::mat E_ = opencv2arma(E);

  // svd to get U, V, W
  arma::mat U, V;
  arma::vec s;
  arma::mat W = arma::reshape(arma::mat({
        0, -1, 0,
        1, 0, 0,
        0, 0, 1 }), 3, 3).t();
  svd(U, s, V, E_);

  // four fold ambiguous
  arma::mat u3 = U.col(2);
  vector<arma::mat> p_primes;
  p_primes.push_back(arma::join_rows(U * W * V.t(), u3));
  p_primes.push_back(arma::join_rows(U * W * V.t(), -u3));
  p_primes.push_back(arma::join_rows(U * W.t() * V.t(), u3));
  p_primes.push_back(arma::join_rows(U * W.t() * V.t(), -u3));

  // find the essential matrix and get the rotation and translation
  Mat mask;
  recoverPose(E, pts1, pts2, R, t, focal, pp, mask);

  // rectify to get the perspective projection
  Mat R1, R2, P1, P2, Q;
  stereoRectify(mtx, Mat::zeros(5, 1, CV_64F), mtx, Mat::zeros(5, 1, CV_64F), leftImage.size(), R, t, R1, R2, P1, P2, Q);

  pts3d.clear();
  Mat pts3;

  // temp
  p_primes.clear();

//  for (arma::mat &Pp : p_primes) {
//    cout << Pp << endl << endl;

    // triangulate points with the perspective projection and convert them
//    P1 = arma2opencv(arma::join_rows(arma::eye<arma::mat>(3, 3), arma::vec({ 0, 0, 0 })));
//    P2 = arma2opencv(Pp);
    Mat hpts;
    triangulatePoints(P1, P2, pts1, pts2, hpts);
//    cout << hpts << endl;
    convertPointsFromHomogeneous(hpts.t(), pts3);
    pts3 = kclusterFilter(pts3, 3);
    cout << pts3 << endl;
    pts3d.push_back(pts3);
//  }


  // cluster filter to remove ambiguities from a single object
  //pts3d = kclusterFilter(pts3d, 3);

  // create a stereo matcher and get a depth frame
  /*Ptr<StereoBM> stereo = StereoBM::create();
  Mat disparity;
  stereo->compute(leftImage, rightImage, disparity);
  Mat depth;
  reprojectImageTo3D(disparity, depth, Q, false, CV_32F);*/

  return Mat();
}

void stereoReconstructSparse(vector<string> images) {
  vector<CameraParams> cameras;
  vector<Mat> rot, trans;
  vector< vector<Point2f> > points1, points2;

  rot.push_back(Mat::eye(3, 3, CV_64F));
  trans.push_back(Mat::zeros(3, 1, CV_64F));

  for (int i = 1; i < images.size(); i++) {
    Mat leftimage = imread(images[i-1]);
    Mat rightimage = imread(images[i]);

    Mat r, t;
    vector<Mat> hpts;
    vector<Point2f> pts1, pts2;
    Mat depth = depth2(leftimage, rightimage, r, t, hpts, pts1, pts2);

    // push back the data
    rot.push_back(r);
    trans.push_back(t);
  }

  // filter out irrelevant matches (to do)
  //
  // bundle adjust using cvsba
  vector<Point3d> points3d;
  vector< vector<Point2d> > pointsImage;
  vector< vector<int> > visibility;
  
}

int main(int argc, char *argv[]) {
  if (argc < 3) {
    printf("usage: %s [img1name] [img2name]\n", argv[0]);
    return 1;
  }
  srand(getpid());

  arma::mat K = reshape(arma::mat({
        545.82463365389708, 0, 319.5,
        0, 545.82463365389708, 2.395,
        0, 0, 1 }), 3, 3).t();
  arma::mat D = arma::mat({ -0.17081096154528716, 0.26412699622915992,
      0, 0, -0.080381316677811496 }).t();
  mtx = arma2opencv(K);
  dist = arma2opencv(D);

  string limgname = string(argv[1]);
  string rimgname = string(argv[2]);

  Mat leftImage = imread(limgname, IMREAD_GRAYSCALE);
  Mat rightImage = imread(rimgname, IMREAD_GRAYSCALE);

  Mat R, t;
  vector<Mat> pts3d;
  vector<Point2f> pts1, pts2;
  Mat newleft, newright;
  undistort(leftImage, newleft, mtx, dist);
  undistort(rightImage, newright, mtx, dist);
  Mat depth = depth2(newleft, newright, R, t, pts3d, pts1, pts2);

  imshow("left", newleft);
  imshow("right", newright);

  imshow("depth", depth);
  waitKey(0);
  imwrite("depth.png", depth);

  return 0;
}
