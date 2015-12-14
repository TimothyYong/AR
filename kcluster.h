#ifndef __kcluster_h__
#define __kcluster_h__

#include <armadillo>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

cv::Mat kclusterFilter(const cv::Mat &S, int k);

#endif
