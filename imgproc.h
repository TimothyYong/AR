#ifndef __TK_IMGPROC_H__
#define __TK_IMGPROC_H__

#include <armadillo>

/** Resize an image using bilinear interpolation
 *  @param A the image to resize
 *  @param m the number of rows of the new size
 *  @param n the number of cols of the new size
 *  @return the interpolated image
 */
arma::mat imresize2(const arma::mat &A, arma::uword m, arma::uword n);
arma::cube imresize2(const arma::cube &C, arma::uword m, arma::uword n);

#endif
