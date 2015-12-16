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

void grabFeatures(Mat img, vector<KeyPoint> &kp, Mat &des) {
  Ptr<SIFT> sift = SIFT::create();
  sift->detectAndCompute(img, noArray(), kp, des);
}

Mat filterZ(Mat pts3d, vector<Point2f> &pts1, vector<Point2f> &pts2, vector<DMatch> &good) {
  arma::mat _pts3d = opencv2arma(pts3d);
  vector<Point3f> pts;
  vector<Point2f> npts1, npts2;
  vector<DMatch> ngood;
  for (int i = 0; i < (int)_pts3d.n_rows; i++) {
    if (_pts3d(i, 2) > 0) {
      pts.push_back(pts3d.at<Point3f>(i, 0));
      npts1.push_back(pts1[i]);
      npts2.push_back(pts2[i]);
      ngood.push_back(good[i]);
    }
  }
  pts1 = npts1;
  pts2 = npts2;
  good = ngood;
  Mat fz(pts.size(), 1, CV_32FC3);
  for (int i = 0; i < pts.size(); i++) {
    fz.at<Point3f>(i, 0) = pts[i];
  }
  return fz;
}

int testTriangulate(Mat &pts3d) {
  arma::mat _pts3d = opencv2arma(pts3d);
  int numz = 0;
  for (int i = 0; i < (int)_pts3d.n_rows; i++) {
    if (_pts3d(i, 2) > 0) {
      numz++;
    }
  }
  cout << "tt" << endl;
  cout << numz << endl;
  return numz;
}

Mat depth2(const Mat leftImage, const Mat rightImage, const vector<KeyPoint> kp1, const vector<KeyPoint> kp2, const Mat des1, const Mat des2,
    Mat &R, Mat &t, Mat &pts3d, vector<Point2f> &pts1, vector<Point2f> &pts2, vector<DMatch> &good) {
  // focal and point projective
  Point2d pp(mtx.at<double>(0, 2), mtx.at<double>(1, 2));
  double focal = mtx.at<double>(0, 0);

  // FLANN parameters
  Ptr<IndexParams> indexParams = makePtr<KDTreeIndexParams>(5);
  Ptr<SearchParams> searchParams = makePtr<SearchParams>(50);
  FlannBasedMatcher flann(indexParams, searchParams);
  vector< vector<DMatch> > matches;
  flann.knnMatch(des1, des2, matches, 2);

  // ratio test (as per Lowe's paper)
  good.clear();
  pts1.clear();
  pts2.clear();
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
  imshow("matches", matchFrame);
  waitKey(0);
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
  arma::mat u3 = U.col(2);

  // 4-fold ambiguity
  vector<arma::mat> p_primes;
  p_primes.push_back(arma::join_rows(U * W * V.t(), u3));
  p_primes.push_back(arma::join_rows(U * W * V.t(), -u3));
  p_primes.push_back(arma::join_rows(U * W.t() * V.t(), u3));
  p_primes.push_back(arma::join_rows(U * W.t() * V.t(), -u3));

  // find the essential matrix and get the rotation and translation
  Mat mask;
  recoverPose(E, pts1, pts2, R, t, focal, pp, mask);

  // match up the matrix to the recover pose to get the best possible solution
  /*Mat Q;
  int maxz = 0;
  for (int i = 0; i < 4; i++) {
    maxz = 0;
    arma::mat P_ = p_primes[i];
    arma::mat R_ = P_(arma::span::all, arma::span(0, 2));
    arma::mat t_ = P_.col(3);
    Mat tempR = arma2opencv(R_);
    Mat tempt = arma2opencv(t_);

    // rectify to get the perspective projection
    Mat R1, R2, P1, P2, tQ;
    stereoRectify(mtx, Mat::zeros(5, 1, CV_64F), mtx, Mat::zeros(5, 1, CV_64F), leftImage.size(), tempR, tempt, R1, R2, P1, P2, tQ);

    // triangulate and cluster
    Mat hpts, dpts;
    triangulatePoints(P1, P2, pts1, pts2, hpts);
    convertPointsFromHomogeneous(hpts.t(), dpts);

    // once triangulate found, test it out, and set if better
    int interimz = 0;
    if ((interimz = testTriangulate(dpts)) > maxz) {
      maxz = interimz;
      pts3d = dpts;
      R = tempR;
      t = tempt;
      Q = tQ;
    }
  }*/

  Mat R1, R2, P1, P2, Q;
  stereoRectify(mtx, Mat::zeros(5, 1, CV_64F), mtx, Mat::zeros(5, 1, CV_64F), leftImage.size(), R, t, R1, R2, P1, P2, Q);
  Mat hpts, dpts;
  triangulatePoints(P1, P2, pts1, pts2, hpts);
  convertPointsFromHomogeneous(hpts.t(), dpts);
  pts3d = dpts;

  // filter out the negative z's
  //pts3d = filterZ(pts3d, pts1, pts2, good);
  // filter out bad clusters
  //pts3d = kclusterFilter(pts3d, 2);

  // create a stereo matcher and get a depth frame
  Ptr<StereoBM> stereo = StereoBM::create();
  Mat disparity;
  stereo->compute(leftImage, rightImage, disparity);
  Mat depth;
  reprojectImageTo3D(disparity, depth, Q, false, CV_32F);

  return depth;
}

vector<Point3d> projectEverything(vector<Mat> rot, vector<Mat> trans, vector<Point3d> points3d, vector<int> camid) {
  assert(rot.size() == trans.size());
  vector<Point3d> newpts;
  Mat lastR;
  Mat lastT;
  for (int j = 0; j < points3d.size(); j++) {
    lastR = rot[0];
    lastT = trans[0];
    for (int i = 0; i < camid[j]; i++) {
      Mat rvec3, tvec3;
      composeRT(lastR, lastT, rot[i], trans[i], rvec3, tvec3);
      lastR = rvec3;
      lastT = tvec3;
    }
    Mat _, newpt;
    Mat temptrans(3, 1, CV_64F);
    temptrans.at<double>(0, 0) = points3d[j].x;
    temptrans.at<double>(1, 0) = points3d[j].y;
    temptrans.at<double>(2, 0) = points3d[j].z;
    composeRT(lastR, lastT, Mat::zeros(3, 1, CV_64F), temptrans, _, newpt);
    newpts.push_back(Point3d(newpt.at<double>(0, 0), newpt.at<double>(1, 0), newpt.at<double>(2, 0)));
  }
  assert(newpts.size() == points3d.size());
  return newpts;
}

void draw3d(Mat &img, vector<Point2f> pts) {
  for (Point2f p: pts) {
    if (p.x >= 0 && p.x < img.cols &&
        p.y >= 0 && p.y < img.rows) {
      img.at<unsigned char>(p.y, p.x) = 255;
    }
  }
}

Mat draw(Mat img, vector<Point2f> imagePoints) {
//  for (int i = 0; i < imagePoints.size(); i++) {
//    circle(img, imagePoints[i], 4, Scalar(255, 0, 0));
//  }
  
  // draw lines
  for (int i = 0; i < 4; i++) {
    line(img, imagePoints[i], imagePoints[i+4], Scalar(255), 2);
  }
  for (int i = 0; i < 4; i++) {
    line(img, imagePoints[i], imagePoints[(i+1)%4], Scalar(255), 2);
  }
  for (int i = 0; i < 4; i++) {
    line(img, imagePoints[i+4], imagePoints[(i+1)%4+4], Scalar(255), 2);
  }
  return img;
}

vector<Mat> stereoReconstructSparse(vector<string> images) {
  vector<Mat> rot, trans, cameraMatrix, distCoeffs;
  vector< vector<int> > idx(images.size());
  vector<Point3d> points3d;
  vector< vector<Point2d> > imagePoints(images.size());
  vector< vector<int> > visibility(images.size());
  vector< vector<Point2f> > points1(images.size()), points2(images.size());
  vector<int> camid;

  vector<Mat> scene;
  vector< vector<KeyPoint> > keypoints;
  vector<Mat> descriptors;
  vector<DMatch> matches;

  // init parameters
  Mat lastR;
  Rodrigues(Mat::eye(3, 3, CV_64F), lastR);
  rot.push_back(lastR);
  Mat lastT = Mat::zeros(3, 1, CV_64F);
  trans.push_back(lastT);
  cameraMatrix.push_back(mtx);
  distCoeffs.push_back(Mat::zeros(5, 1, CV_64F));

  // get the calibration parameters for the function
  for (int i = 0; i < images.size(); i++) {
    cout << "Attempting to read: " << images[i] << endl;
    Mat img = imread(images[i], IMREAD_GRAYSCALE), uimg;
    undistort(img, uimg, mtx, dist);
    scene.push_back(uimg);
    vector<KeyPoint> kp;
    Mat des;
    grabFeatures(img, kp, des);
    keypoints.push_back(kp);
    descriptors.push_back(des);
  }

  // get the depth points from the triangulation
  cout << "now trying to triangulate\n";
  for (int i = 1; i < images.size(); i++) { // for every camera
    Mat r, t, pts3d;
    vector<Point2f> pts1, pts2;
    vector<KeyPoint> kp1 = keypoints[i-1];
    vector<KeyPoint> kp2 = keypoints[i];
    Mat des1 = descriptors[i-1];
    Mat des2 = descriptors[i];
    Mat leftimage = scene[i-1];
    Mat rightimage = scene[i];
    Mat depth = depth2(leftimage, rightimage, kp1, kp2, des1, des2, r, t, pts3d, pts1, pts2, matches);
    points1[i] = pts1;
    points2[i] = pts2;

    // push back the data
    int N = pts3d.rows;
    for (int j = 0; j < N; j++) { // for every datpoint taken from the camera
      // find location of previous match
      DMatch match = matches[j];
      int matchidx = -1;
      for (int k = 0; k < idx[0].size(); k++) {
        if (idx[i-1][k] == match.queryIdx) {
          matchidx = k;
          break;
        }
      }
      // place the image points and the visibility in the matrix
      if (matchidx != -1) {
        imagePoints[i][matchidx] = Point2d(pts2[j].x, pts2[j].y);
        visibility[i][matchidx] = 1;
        idx[i][matchidx] = match.trainIdx;
      } else { 
        // update camera points
        imagePoints[i-1].push_back(Point2d(pts1[j].x, pts1[j].y));
        visibility[i-1].push_back(1);
        idx[i-1].push_back(match.queryIdx);
        imagePoints[i].push_back(Point2d(pts2[j].x, pts2[j].y));
        visibility[i].push_back(1);
        idx[i].push_back(match.trainIdx);
        for (int k = 0; k < images.size(); k++) {
          if (k != i && k != (i-1)) {
            imagePoints[k].push_back(Point2d(0, 0));
            visibility[k].push_back(0);
            idx[k].push_back(-1);
          }
        }
        // place the 3d point in
        Point3f pt = pts3d.at<Point3f>(j, 0);
        points3d.push_back(Point3d(pt.x, pt.y, pt.z));
        camid.push_back(i-1);
      }
    }
    // place the camera matrix, the rotation, the translation, and the distortion in
    cameraMatrix.push_back(mtx);
    Mat r3, rvec3, tvec3;
    Rodrigues(r, r3);
    composeRT(lastR, lastT, r3, t, rvec3, tvec3); // rotate the vectors
    lastR = rvec3;
    lastT = tvec3;
    rot.push_back(lastR);
    trans.push_back(lastT);
    distCoeffs.push_back(Mat::zeros(5, 1, CV_64F));
  }

  vector<Point3d> projection = projectEverything(rot, trans, points3d, camid);
  for (int i = 0; i < projection.size(); i++) {
    if (i == 0) {
      cout << "[";
    }
    printf("%lf, %lf, %lf", projection[i].x, projection[i].y, projection[i].z);
    if (i == projection.size()-1) {
      cout << "]\n";
    } else {
      cout << ";\n";
    }
  }

  // bundle adjust using cvsba
  cout << "bundle adjusting\n";
  printf("cam check: %zd %zd\n", imagePoints.size(), visibility.size());
  printf("check: %zd %zd %zd %zd %zd %zd %zd\n", points3d.size(), imagePoints[0].size(), visibility[0].size(), cameraMatrix.size(), rot.size(), trans.size(), distCoeffs.size());
  printf("check2: %d %d %d %d\n", rot[0].rows, rot[0].cols, trans[0].rows, trans[0].cols);
  for (int i = 0; i < rot.size(); i++) {
    cout << "R and t" << endl;
    cv::Mat rr = rot[i];
    cv::Mat tt = trans[i];
    cout << rr << endl << tt << endl;
  }
  cvsba::Sba sba;
  sba.run(points3d, imagePoints, visibility, cameraMatrix, rot, trans, distCoeffs);
  cout << "Initial error: " << sba.getInitialReprjError() << endl <<
          "Final error: " << sba.getFinalReprjError() << endl;

  // reproject the points (sparse)
  projection = projectEverything(rot, trans, points3d, camid);
  for (int i = 0; i < projection.size(); i++) {
    if (i == 0) {
      cout << "[";
    }
    printf("%lf, %lf, %lf", projection[i].x, projection[i].y, projection[i].z);
    if (i == projection.size()-1) {
      cout << "]\n";
    } else {
      cout << ";\n";
    }
  }
  vector<Point3f> pts3f;
  for (Point3d p : points3d) {
    pts3f.push_back(Point3f(p.x, p.y, p.z));
  }
  vector<Mat> ppps;
  Mat ppimg;
  char buf[256];
  for (int i = 0; i < 3; i++) {
    ppimg = Mat::zeros(480, 640, CV_8UC1);
    vector<Point2f> ppp;
    projectPoints(pts3f, rot[i], trans[i], mtx, Mat::zeros(5,1,CV_64F), ppp);
    draw3d(ppimg, ppp);
    sprintf(buf, "recon%d.png", i);
    imshow(buf, ppimg);
    imwrite(buf, ppimg);
  }
  waitKey(0);

  // find center
  Point3f avg(0, 0, 0);
  for (int i = 0; i < pts3f.size(); i++) {
    avg += pts3f[i];
  }
  avg.x /= pts3f.size();
  avg.y /= pts3f.size();
  avg.z /= pts3f.size();
  
  vector<Point3f> axis;
  double s = 80;
      axis.push_back(Point3f(0, 0, 0)+avg);
      axis.push_back(Point3f(0, s, 0)+avg);
      axis.push_back(Point3f(s, s, 0)+avg);
      axis.push_back(Point3f(s, 0, 0)+avg);
      axis.push_back(Point3f(0, 0, -s)+avg);
      axis.push_back(Point3f(0, s, -s)+avg);
      axis.push_back(Point3f(s, s, -s)+avg);
      axis.push_back(Point3f(s, 0, -s)+avg);
  vector<Point2f> imgPoints;

  Mat nK = mtx;
  //nK.at<double>(0, 2) = 5000;
  //nK.at<double>(1, 2) = 5000;

  // project 3d AR
  for (int i = 0; i < 2; i++) {
    imgPoints.clear();
    ppimg = Mat::zeros(480, 640, CV_8UC1);
    vector<Point2f> ppp;
    projectPoints(pts3f, rot[i], trans[i]+arma2opencv(arma::reshape(arma::mat({0, 0, 500}),3,1)), nK, Mat::zeros(5,1,CV_64F), ppp);
    draw3d(ppimg, ppp);
    sprintf(buf, "ar_recon%d.png", i);
    projectPoints(axis, rot[i], trans[i]+arma2opencv(arma::reshape(arma::mat({0, 0, 500}),3,1)), nK, Mat::zeros(5,1,CV_64F), imgPoints);
    ppimg = draw(ppimg, imgPoints);
    imshow(buf, ppimg);
    imwrite(buf, ppimg);
  }
  waitKey(0);

  // EC: project the FACE
  for (int i = 0; i < 2; i++) {
    imgPoints.clear();
    ppimg = Mat::zeros(480, 640, CV_8UC1);
    vector<Point2f> ppp;
    projectPoints(pts3f, rot[i], trans[i]+arma2opencv(arma::reshape(arma::mat({0, 0, 500}),3,1)), nK, Mat::zeros(5,1,CV_64F), ppp);
    draw3d(ppimg, ppp);
    sprintf(buf, "arface_recon%d.png", i);
    projectPoints(axis, rot[i], trans[i]+arma2opencv(arma::reshape(arma::mat({0, 0, 500}),3,1)), nK, Mat::zeros(5,1,CV_64F), imgPoints);
    ppimg = draw(ppimg, imgPoints);

    Mat imagey = imread("one-punch.jpg", IMREAD_GRAYSCALE);
      vector<Point2f> src = { Point2f(0, 0), Point2f(0, imagey.rows), Point2f(imagey.cols, 0), Point2f(imagey.cols, imagey.rows) };
      vector<Point2f> dst = { imgPoints[4], imgPoints[5], imgPoints[7], imgPoints[6] };
      Mat hy = findHomography(src, dst);
      Mat subimagey;
      warpPerspective(imagey, subimagey, hy, ppimg.size());
      Mat mask = Mat::ones(imagey.rows, imagey.cols, CV_8UC1);
      Mat submask;
      warpPerspective(mask, submask, hy, ppimg.size());
      subimagey.copyTo(ppimg, submask);

    imshow(buf, ppimg);
    imwrite(buf, ppimg);
  }
  waitKey(0);

  return ppps;
}

int main(int argc, char *argv[]) {
  srand(getpid());

  arma::mat K = reshape(arma::mat({
      545.82463365389708, 0, 319.5,
      0, 545.82463365389708, 239.5,
      0, 0, 1 }), 3, 3).t();
  arma::mat D = arma::mat({ -0.17081096154528716, 0.26412699622915992,
      0, 0, -0.080381316677811496 }).t();
  mtx = arma2opencv(K);
  dist = arma2opencv(D);

  vector<string> names;
  for (int i = 0; i < 7; i++) {
    char imgname[256];
    sprintf(imgname, "jumper/img%02d.png", i);
    names.push_back(string(imgname));
  }

  vector<Mat> recon = stereoReconstructSparse(names);
  waitKey(0);

  return 0;
}
