#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <cstdio>
#include <iostream>

using namespace cv;
using namespace std;

int main(int argc, char *argv[]) {
  Mat img;
  if (argc < 2) {
    printf("usage: %s [prefix]\n", argv[0]);
    return 1;
  }
  VideoCapture cam(0);

  int count = 0;
  char fname[256];
  while (count < 30) {
    cam >> img;

    // show the images
    imshow("camera", img);

    if (waitKey(30) >= 0) {
      printf("reading image %d\n", count);
      // save image
      sprintf(fname, "%simg%02d.png", argv[1], count);
      imwrite(fname, img);
      count++;
    }
  }
  return 0;
}
