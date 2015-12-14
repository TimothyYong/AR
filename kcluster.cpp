#include "kcluster.h"
#include "highgui.h"
#include <vector>

using namespace std;
using namespace arma;

cv::Mat kclusterFilter(const cv::Mat &S, int k) {
  assert(k < S.rows && k > 0);
  assert(S.rows > 1 && S.cols == 1);
  // create centroids
  mat S_ = opencv2arma(S).t();
  vector<vec> centroids;
  for (int i = 0; i < k ; i++) {
    centroids.push_back(S_.col(rand() % S_.n_cols));
  }

  // cluster 15x
  vector< vector<vec> > part;
  for (int _i = 0; _i < 15; _i++) {
    // assign
    part.clear();
    for (int i = 0; i < k; i++) {
      part.push_back(vector<vec>());
    }
    for (int i = 0; i < S_.n_cols; i++) {
      vec diff = centroids[0] - S_.col(i);
      double min_val = sqrt(dot(diff, diff));
      int min_ind = 0;
      for (int j = 0; j < centroids.size(); j++) {
        diff = centroids[j] - S_.col(i);
        double interim = sqrt(dot(diff, diff));
        if (interim < min_val) {
          min_val = interim;
          min_ind = j;
        }
      }
      part[min_ind].push_back(S_.col(i));
    }
    // update
    for (int i = 0; i < k; i++) {
      if (part[i].size() > 0) {
        vec summation(S_.n_rows, fill::zeros);
        for (vec &p : part[i]) {
          summation += p;
        }
        centroids[i] = summation / (double)part[i].size();
      }
    }
  }

  // select the highest one
  int max_ind = 0;
  for (int i = 0; i < centroids.size(); i++) {
    if (part[i].size() > part[max_ind].size()) {
      max_ind = i;
    }
  }

  vector<vec> mp = part[max_ind];
  cv::Mat cluster(mp.size(), 1, CV_32FC3);
  for (int j = 0; j < mp.size(); j++) {
    cluster.at<cv::Vec3f>(j, 0) = 
      cv::Vec3f(mp[j](0), mp[j](1), mp[j](2));
  }

  return cluster;
}
