"""
    Contains image processing algorithms which may be used on their
    own, or as smaller building blocks for a composite algorithm.
"""

import cv2
import numpy as np
from numpy.linalg import norm, svd

import swiftwatcher_refactor.interface.video_io as vio
from scipy import ndimage
from skimage import measure


###############################################################################
#                  CROPPING/ROI REGION FUNCTIONS BEGIN HERE                   #
###############################################################################


def generate_regions(filepath, corners):
    """Generate various regions of interest used by the swift counting
    algorithm to crop, resize, and detect events."""

    resize_dim = (300, 150)
    crop_region = generate_crop_region(corners)
    roi_mask = generate_roi_mask(filepath, corners, crop_region, resize_dim)

    return crop_region, roi_mask, resize_dim


def generate_crop_region(corners):
    """Generate rectangular region from two corners of chimney:
                        _______________________
                       |                       |
                       |      CROP REGION      |
                       |  *-----------------*  |
                       |__|_________________|__|
                          |                 |
                          |  chimney stack  |

    Region will be used to crop frame and isolate the edge of the
    chimney stack."""

    # From provided corners, determine which coordinates are the
    # outer-most ones.
    left, right, bottom = determine_chimney_extents(corners)

    # Dimensions = (1.25 X 0.625) = (2 X 1) ratio of width to height
    width = right - left
    crop_region = [(left - int(0.125 * width), bottom - int(0.5 * width)),
                   (right + int(0.125 * width), bottom + int(0.125 * width))]

    return crop_region


def generate_roi_crop_region(corners):
    """Generate rectangular regions from two corners of chimney:
                            ________________
                           |   ROI REGION  |
                          *-----------------*
                          |                 |
                          |                 |
                          |  chimney stack  |

    Region will be used when generating the chimney's ROI, which
    helps to flag when a chimney swift may have entered a chimney"""

    left, right, bottom = determine_chimney_extents(corners)

    # Left and right brought in slightly as swifts don't enter at edge
    width = right - left
    roi_region = [(int(left + 0.025 * width), int(bottom - 0.25 * width)),
                  (int(right - 0.025 * width), int(bottom))]

    return roi_region


def determine_chimney_extents(corners):
    """Determine the outermost coordinates from the two corners that
    define the chimney.

                 (x1, y1) *-----------------* (x2, y2)
                          |                 |
                          |  chimney stack  |
                          |                 |                        """

    left = min(corners[0][0], corners[1][0])
    right = max(corners[0][0], corners[1][0])
    bottom = max(corners[0][1], corners[1][1])

    return left, right, bottom


###############################################################################
#                        ROI MASK FUNCTIONS BEGIN HERE                        #
###############################################################################


def generate_roi_mask(filepath, corners, crop_region, resize_dim):
    """Generate a mask that contains the chimney's region of interest."""

    frame = vio.get_first_video_frame(filepath)

    # Create ROI mask using a subregion of the frame
    roi_region = generate_roi_crop_region(corners)
    cropped_frame = crop_frame(frame, roi_region)
    blurred_frame = median_blur(cropped_frame, 9)
    blurred_frame = median_blur(blurred_frame, 9)
    b_channel, _, _ = split_bgr_channels(blurred_frame)
    thresh_frame = threshold_channel(b_channel)
    edge_frame = detect_canny_edges(thresh_frame)
    dilated_frame = dilate_upwards(edge_frame, 20)

    # Take smaller ROI region mask and expand it to be size of full frame
    unprocessed_mask = create_mask(dilated_frame, roi_region, frame)

    # Apply same preprocessing as the frames themselves, then threshold again
    grayscale_mask = convert_grayscale(unprocessed_mask)
    cropped_mask = crop_frame(grayscale_mask, crop_region)
    resized_mask = resize_frame(cropped_mask, resize_dim)
    roi_mask = threshold_channel(resized_mask)

    return roi_mask


def median_blur(image, kernel_size):
    """Apply OpenCV's median blur to image."""

    blurred_image = cv2.medianBlur(image, kernel_size)

    return blurred_image


def split_bgr_channels(image):
    """Split 3-channel BGR image into individual channels."""

    b, g, r = cv2.split(image)

    return b, g, r


def threshold_channel(image):
    """Apply Otsu's automatic thresholding algorithm to a 1-channel
    input image. Using this on the B channel of a BGR image was found to
    work well at thresholding a chimney from a background of sky."""

    _, thresholded_image = cv2.threshold(image, 0, 255,
                                         cv2.THRESH_BINARY +
                                         cv2.THRESH_OTSU)

    return thresholded_image


def detect_canny_edges(image):
    """Find the edges of an input image using Canny edge detection."""

    edge_image = cv2.Canny(image, 0, 256)

    return edge_image


def dilate_upwards(image, N):
    """Apply dilation using an Nx1 kernel, with anchor coordinate such
    that dilation only occurs in upwards direction."""

    dilated_image = cv2.dilate(image,
                               kernel=np.ones((N, 1), np.uint8),
                               anchor=(0, 0))

    return dilated_image


def create_mask(mask, frame_region, frame):
    """Take smaller ROI mask (subregion in image) and insert it into
    a blank image the size of the full frame."""

    frame_with_mask = np.zeros_like(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    frame_with_mask[frame_region[0][1]:frame_region[1][1],
                    frame_region[0][0]:frame_region[1][0]] = mask

    return frame_with_mask


###############################################################################
#                     PREPROCESSING FUNCTIONS BEGIN HERE                      #
###############################################################################


def convert_grayscale(frame):
    """Convert a frame from 3-channel RGB to grayscale."""

    if len(frame.shape) is 3:
        grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif len(frame.shape) is 2:
        grayscale_frame = frame

    return grayscale_frame


def crop_frame(frame, crop_region):
    """Crop frame to dimensions specified by generate_crop_region."""

    return frame[crop_region[0][1]:crop_region[1][1],
                 crop_region[0][0]:crop_region[1][0]]


def resize_frame(frame, dimensions):
    """Resize frame so dimensions are fixed regardless of chimney
    size. (Different chimneys produce different crop dimensions.)"""

    resized_frame = cv2.resize(frame, dimensions)

    return resized_frame


###############################################################################
#                      SEGMENTATION FUNCTIONS BEGIN HERE                      #
###############################################################################


def rpca(frame_list):
    """Decompose set of images into corresponding low-rank and sparse
    images. Method expects images to have been reshaped to matrix of
    column vectors.

    Note: frame = lowrank + sparse, where:
                  lowrank = "background" image
                  sparse  = "foreground" errors corrupting
                            the "background" image

    The size of the queue will determine the tradeoff between
    computational efficiency and accuracy."""

    # Reshape frames into column vector matrix, 1 vector for each frame
    img_matrix = np.array(frame_list)
    col_matrix = np.transpose(img_matrix.reshape(img_matrix.shape[0],
                                                 img_matrix.shape[1] *
                                                 img_matrix.shape[2]))

    # Algorithm for the IALM approximation of Robust PCA method.
    lr_columns, s_columns = \
        inexact_augmented_lagrange_multiplier(col_matrix)

    # Bring pixels that are darker than background to [0, 255] range
    s_columns = np.negative(s_columns)
    s_columns = np.clip(s_columns, 0, 255).astype(np.uint8)

    # Reshape columns back into image dimensions and store in list
    output_frames = [np.reshape(s_columns[:, i],
                                (img_matrix.shape[1],
                                 img_matrix.shape[2]))
                     for i in range(img_matrix.shape[0])]

    return output_frames


def inexact_augmented_lagrange_multiplier(X, lmbda=0.01, tol=0.001,
                                          maxiter=100, verbose=False):
    """Inexact Augmented Lagrange Multiplier algorithm for Robust PCA.
    matrix decomposition. Decomposes an input matrix X into a
    low-rank approximation and sparse components.

    Can be used for background subtraction in image sequences if images
    are shaped into column vectors of X.

    Implementation borrowed directly from:
        https://github.com/kastnerkyle"""

    Y = X
    norm_two = norm(Y.ravel(), 2)
    norm_inf = norm(Y.ravel(), np.inf) / lmbda
    dual_norm = np.max([norm_two, norm_inf])
    Y = Y / dual_norm
    A = np.zeros(Y.shape)
    E = np.zeros(Y.shape)
    dnorm = norm(X, 'fro')
    mu = 1.25 / norm_two
    rho = 1.5
    sv = 10.
    n = Y.shape[0]
    itr = 0
    while True:
        Eraw = X - A + (1 / mu) * Y
        Eupdate = np.maximum(Eraw - lmbda / mu, 0) + np.minimum(Eraw + lmbda / mu, 0)
        U, S, V = svd(X - Eupdate + (1 / mu) * Y, full_matrices=False)
        svp = (S > 1 / mu).shape[0]
        if svp < sv:
            sv = np.min([svp + 1, n])
        else:
            sv = np.min([svp + round(.05 * n), n])
        Aupdate = np.dot(np.dot(U[:, :svp], np.diag(S[:svp] - 1 / mu)), V[:svp, :])
        A = Aupdate
        E = Eupdate
        Z = X - A - E
        Y = Y + mu * Z
        mu = np.min([mu * rho, mu * 1e7])
        itr += 1
        if ((norm(Z, 'fro') / dnorm) < tol) or (itr >= maxiter):
            break
    if verbose:
        print("Finished at iteration %d" % (itr))
    return A, E


def bilateral_blur(frame, d, sigmaColor, sigmaSpace):
    blurred_frame = cv2.bilateralFilter(frame, d, sigmaColor, sigmaSpace)

    return blurred_frame.astype(np.uint8)


def thresh_to_zero(frame, thresh):
    _, thresholded_frame = cv2.threshold(frame,
                                         thresh=thresh,
                                         maxval=255,
                                         type=cv2.THRESH_TOZERO)

    return thresholded_frame.astype(np.uint8)


def grayscale_opening(frame, SE):
    opened_frame = ndimage.grey_opening(frame, size=SE)

    return opened_frame.astype(np.uint8)


def cc_labeling(frame, connectivity):
    # Segment using CC labeling
    _, labeled_frame = cv2.connectedComponents(frame, connectivity)

    return labeled_frame.astype(np.uint8)


def get_segment_properties(frame):

    # "coordinates='xy'" suppresses a warning found in skimage 0.15, See:
    # https://scikit-image.org/docs/0.15.x/release_notes_and_installation.html
    return measure.regionprops(frame, coordinates='xy')
