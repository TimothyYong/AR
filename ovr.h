#ifndef __OVR_H__
#define __OVR_H__

#include <armadillo>

/** Create an oculus rift image from two different images
 *  @param left_image the left image to barrellize
 *  @param right_image the right image to barrellize
 *  @param offset_x offset for the barrel center
 *  @return the merge of both images
 */
arma::cube ovr_image(const arma::cube &left, const arma::cube &right, double offset_x = 0.15);


#endif
