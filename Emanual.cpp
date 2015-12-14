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
  Mat F = findFundamentalMat(pts1, pts2, FM_RANSAC, 3.0, 0.99);
  arma::mat F_ = opencv2arma(F);
  arma::mat K_ = opencv2arma(mtx);
  arma::mat E_ = K_.t() * F_ * K_;

  // or use the built in function (easy way)
  //Mat E = findEssentialMat(pts1, pts2, focal, pp);

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
  arma::mat P_1 = arma::join_rows(U * W * V.t(), u3);
  arma::mat P_2 = arma::join_rows(U * W * V.t(), -u3);
  arma::mat P_3 = arma::join_rows(U * W.t() * V.t(), u3);
  arma::mat P_4 = arma::join_rows(U * W.t() * V.t(), -u3);

  cout << "USV: (SVD)" << endl;
  cout << U << endl << s << endl << V << endl;

  Mat E = arma2opencv(E_);
  cout << "E: " << endl;
  cout << E << endl;

  cout << "possible: " << endl;
  cout << P_1 << endl;
  cout << P_2 << endl;
  cout << P_3 << endl;
  cout << P_4 << endl;
  cout << endl;

  // hack: choose a random matrix out of all 4
  arma::mat P_prime;
  int index = rand() % 4;
  for (int i = 0; i < 4; i++) {
    switch (i) {
      case 0:
        P_prime = P_1;
        break;
      case 1:
        P_prime = P_2;
        break;
      case 2:
        P_prime = P_3;
        break;
      case 3:
        P_prime = P_4;
        break;
    }
  }
  // find the essential matrix and get the rotation and translation (easy way)
  //Mat mask;
  //recoverPose(E, pts1, pts2, R, t, focal, pp, mask);

  // or do it H&Z way
  arma::mat R_ = P_prime.cols(0, 2);
  arma::mat t_ = P_prime.col(3);

  R = arma2opencv(R_);
  t = arma2opencv(t_);

  cout << "r and t" << endl;
  cout << R << endl;
  cout << t << endl;

  // rectify to get the perspective projection
  Mat R1, R2, P1, P2, Q;
  stereoRectify(mtx, Mat::zeros(5, 1, CV_64F), mtx, Mat::zeros(5, 1, CV_64F), leftImage.size(), R, t, R1, R2, P1, P2, Q);

  // triangulate points with the perspective projection
  Mat hpts;
  triangulatePoints(P1, P2, pts1, pts2, hpts);
  convertPointsFromHomogeneous(hpts.t(), pts3d);

  // filter out the ambiguities (assumption: local cluster ONLY)
  pts3d = kclusterFilter(pts3d, 3); // special method

  // create a stereo matcher and get a depth frame
  Ptr<StereoBM> stereo = StereoBM::create();
  Mat disparity;
  stereo->compute(leftImage, rightImage, disparity);
  Mat depth;
  reprojectImageTo3D(disparity, depth, Q, false, CV_32F);

  imshow(name, depth);
  waitKey(0);
  return depth;
}
