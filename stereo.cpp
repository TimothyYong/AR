#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/line_descriptor/descriptor.hpp>
#include <opencv2/stitching/detail/motion_estimators.hpp>
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
const int nmatches = 35;

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
  arma::mat bad_match(disparity.n_rows, disparity.n_cols, arma::fill::zeros);
  arma::mat depth(disparity.n_rows, disparity.n_cols, arma::fill::zeros);
  disparity /= disparity.max(); // normalize
  for (arma::uword i = 0; i < disparity.n_rows; i++) {
    for (arma::uword j = 0; j < disparity.n_cols; j++) {
      if (disparity(i, j) == 0.0) {
        depth(i, j) = 0.0;
        bad_match(i, j) = 1.0;
      } else {
        depth(i, j) = 1.0 / disparity(i, j);
        bad_match(i, j) = 0.0;
      }
    }
  }
  depth /= depth.max();
  for (arma::uword i = 0; i < disparity.n_rows; i++) {
    for (arma::uword j = 0; j < disparity.n_cols; j++) {
      if (bad_match(i, j) == 1.0) {
        depth(i, j) = 1.0;
      }
    }
  }
  return depth;
}

Mat depth3(Mat leftImage, Mat rightImage) {
  srand(getpid());

  // Camera Matrix for the PS3 Eye
  Mat mtx(3, 3, CV_64F);
  mtx.at<double>(0, 0) = 545.82463365389708;
  mtx.at<double>(0, 1) = 0;
  mtx.at<double>(0, 2) = 319.5;
  mtx.at<double>(1, 0) = 0;
  mtx.at<double>(1, 1) = 545.82463365389708;
  mtx.at<double>(1, 2) = 2.395;
  mtx.at<double>(2, 0) = 0;
  mtx.at<double>(2, 1) = 0;
  mtx.at<double>(2, 2) = 1;

  // focal and point projective
  Point2d pp(mtx.at<double>(0, 2), mtx.at<double>(1, 2));
  double focal = mtx.at<double>(0, 0);

  // Distortion Coefficients for the PS3 Eye
  Mat dist(5, 1, CV_64F);
  dist.at<double>(0, 0) = -0.17081096154528716;
  dist.at<double>(1, 0) = 0.26412699622915992;
  dist.at<double>(2, 0) = 0;
  dist.at<double>(3, 0) = 0;
  dist.at<double>(4, 0) = -0.080381316677811496;

  namedWindow("leftimage");
  namedWindow("rightimage");
  imshow("leftimage", leftImage);
  imshow("rightimage", rightImage);
  waitKey(0);

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
  vector<Point2f> pts1, pts2;

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

  Mat matchImage = drawMatches(leftImage, kp1, rightImage, kp2, good);
  namedWindow("matches");
  imshow("matches", matchImage);
  waitKey(0);

  /*
  // compute the fundamental matrix (note: change to accompany the instrinsic parameters of the camera)
  // use stereo rectify for that
  Mat fund = findFundamentalMat(pts1, pts2, FM_RANSAC, 3.0, 0.99);
  arma::cube F_ = cvt_opencv2arma(fund);
  arma::mat F = F_.slice(0);

  // find the epilines in both images
  Mat lines1, lines2;
  computeCorrespondEpilines(pts2, 2, fund, lines1);
  computeCorrespondEpilines(pts1, 1, fund, lines2);
  Mat img5, img6;
  drawLines(leftImage, rightImage, lines1, pts1, pts2, img5, img6);
  Mat img3, img4;
  drawLines(rightImage, leftImage, lines2, pts2, pts1, img3, img4);

  namedWindow("leftimage");
  namedWindow("rightimage");
  imshow("leftimage_epi", img5);
  imshow("rightimage_epi", img3);
  waitKey(0);
  */

  /*  vector< vector<Point3f> > objpts(1);
      for (int i = 0; i < pts1.size(); i++) {
      objpts[0].push_back(Point3f((float)(pts1[i].x), (float)(pts1[i].y), 0));
      }
      Mat R, T, E, F;
      double rms = stereoCalibrate(objpts, pts1, pts2, mtx, dist, mtx, dist,
      leftImage.size(), R, T, E, F);
      */

  // find the essential matrix and get the rotation and translation
  Mat E, R, t, mask;
  E = findEssentialMat(pts1, pts2, focal, pp, RANSAC, 0.999, 1.0, noArray());
  recoverPose(E, pts1, pts2, R, t, focal, pp, mask);

  // rectify to get the perspective projection
  Mat R1, R2, P1, P2, Q;
  stereoRectify(mtx, dist, mtx, dist, leftImage.size(), R, t, R1, R2, P1, P2, Q);

  // create a stereo matcher
  Ptr<StereoBM> stereo = StereoBM::create();
  //Ptr<StereoSGBM> stereo = StereoSGBM::create(
  //    16, 96, 16, 8*3*3*3, 32*3*3*3, 1, 10, 100, 32);

  Mat disparity;
  stereo->compute(leftImage, rightImage, disparity);
  Mat depth;
  reprojectImageTo3D(disparity, depth, Q);

  imshow("depth", depth);
  waitKey(0);

  // triangulate points with the perspective projection
  Mat hpts;
  triangulatePoints(P1, P2, pts1, pts2, hpts);

  destroyAllWindows();

  return hpts;
}

void stereoReconstructSparse(vector<string> images) {
  for (int i = 1; i < images.size(); i++) {
    Mat leftimage = imread(images[i-1]);
    Mat rightimage = imread(images[i]);

    // figure out the new camera matrix based on non-linear methods (using the hypothesis mtx)
    CameraParams cam1;
    cam1.ppx = mtx.at<double>(0, 2);
    cam1.ppy = mtx.at<double>(1, 2);
    cam1.focal = focal;
    cam1.R = Mat::eye(3, 3, CV_64F);
    cam1.t = Mat::zeros(3, 1, CV_64F);
    CameraParams cam2;
    cam2.ppx = mtx.at<double>(0, 2);
    cam2.ppy = mtx.at<double>(1, 2);
    cam2.focal = focal;
    cam2.R = R;
    cam2.t = t;
    vector<CameraParams> cameras;
    cameras.push_back(cam1);
    cameras.push_back(cam2);
    BundleAdjusterReproj adjuster;

    Mat hpts = 
  }
}

int main(int argc, char *argv[]) {
  if (argc < 3) {
    printf("usage: %s [img1name] [img2name]\n", argv[0]);
    return 1;
  }

  string limgname = string(argv[1]);
  string rimgname = string(argv[2]);

  Mat leftImage = imread(limgname, IMREAD_GRAYSCALE);
  Mat rightImage = imread(rimgname, IMREAD_GRAYSCALE);

  Mat hpts = depth3(leftImage, rightImage);

  cout << endl;
  for (int j = 0; j < hpts.cols; j++) {
    Mat column = hpts.col(j);
    cout << column << endl;
  }

  return 0;
}
