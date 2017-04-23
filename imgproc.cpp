#include "highgui.h"
#include "imgproc.h"
#include <cstdio>
#include <iostream>

using namespace arma;
using namespace std;

mat imresize2(const mat &A, uword m, uword n) {
  mat G(m, n, fill::zeros);
  double kr = (double)A.n_rows / (double)m;
  double kc = (double)A.n_cols / (double)n;
  // bilinear interpolation
  for (uword i = 0; i < m; i++) {
    for (uword  j = 0; j < n; j++) {
      double y = (double)i * kr - 0.5;
      double x = (double)j * kc - 0.5;
      int i_ = (int)floor(y);
      int j_ = (int)floor(x);
      double dy = 1.0 - (y - floor(y));
      double dx = 1.0 - (x - floor(x));
      double total = 0.0;
      if (j_ >= 0 && j_ < (int)A.n_cols) {
        if (i_ >= 0 && i_ < (int)A.n_rows) {
          G(i, j) += dx * dy * A(i_,j_);
          total += dx * dy;
        }
        if (i_+1 >= 0 && i_+1 < (int)A.n_rows) {
          G(i, j) += dx * (1-dy) * A(i_+1,j_);
          total += dx * (1-dy);
        }
      }
      if (j_+1 >= 0 && j_+1 < (int)A.n_cols) {
        if (i_ >= 0 && i_ < (int)A.n_rows) {
          G(i, j) += (1-dx) * dy * A(i_,j_+1);
          total += (1-dx) * dy;
        }
        if (i_+1 >= 0 && i_+1 < (int)A.n_rows) {
          G(i, j) += (1-dx) * (1-dy) * A(i_+1,j_+1);
          total += (1-dx) * (1-dy);
        }
      }
      if (total != 0.0) {
        G(i, j) /= total;
      } else {
        G(i, j) = 0.0;
      }
    }
  }
  return G;
}

cube imresize2(const cube &C, uword m, uword n) {
  cube F(m, n, C.n_slices);
  for (uword k = 0; k < C.n_slices; k++) {
    F.slice(k) = imresize2(C.slice(k), m, n);
  }
  return F;
}
