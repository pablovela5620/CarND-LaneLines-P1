#importing some useful packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

import math

#reading in an image
solidWhiteRight = mpimg.imread('test_images/solidWhiteRight.jpg')

#printing out some stats and plotting
print('This image is:', type(solidWhiteRight), 'with dimensions:', solidWhiteRight.shape)
plt.xticks(np.arange(0,960,50))
plt.yticks(np.arange(0,540,50))
plt.imshow(solidWhiteRight)  # if you wanted to show a single color channel image called 'gray', for example, call as plt.imshow(gray, cmap='gray')

def grayscale(img):
    """Applies the Grayscale transform
    This will return an image with only one color channel
    but NOTE: to see the returned image as grayscale
    (assuming your grayscaled image is called 'gray')
    you should call plt.imshow(gray, cmap='gray')"""
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Or use BGR2GRAY if you read an image with cv2.imread()
    # return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def canny(img, low_threshold, high_threshold):
    """Applies the Canny transform"""
    return cv2.Canny(img, low_threshold, high_threshold)


def gaussian_blur(img, kernel_size):
    """Applies a Gaussian Noise kernel"""
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)


def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    # defining a blank mask to start with
    mask = np.zeros_like(img)

    # defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    # returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    """
    NOTE: this is the function you might want to use as a starting point once you want to
    average/extrapolate the line segments you detect to map out the full
    extent of the lane (going from the result shown in raw-lines-example.mp4
    to that shown in P1_example.mp4).

    Think about things like separating line segments by their
    slope ((y2-y1)/(x2-x1)) to decide which segments are part of the left
    line vs. the right line.  Then, you can average the position of each of
    the lines and extrapolate to the top and bottom of the lane.

    This function draws `lines` with `color` and `thickness`.
    Lines are drawn on the image inplace (mutates the image).
    If you want to make the lines semi-transparent, think about combining
    this function with the weighted_img() function below
    """
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope=((y2-y1)/(x2-x1))
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

            # x_left=[]
            # y_left=[]
            # x_right=[]
            # y_right=[]
            #
            # for line in lines:
            #     for x1,y1,x2,y2 in line:
            #         slope=((y2-y1)/(x2-x1))
            #         if slope>=0:#Right lane
            #             x_right.extend((x1,x2))
            #             y_right.extend((y1,y2))
            #         elif slope<0 and (x1 <450 and x2<450) :#Left lane
            #             x_left.extend((x1,x2))
            #             y_left.extend((y1,y2))
            #
            # fitR=np.polyfit(x_right,y_right,1)
            # fit_functionR=np.poly1d(fitR)
            # x1R=550
            # y1R=int(fit_functionR(x1R))
            # x2R=850
            # y2R=int(fit_functionR(x2R))
            # cv2.line(img, (x1R, y1R), (x2R, y2R), color, thickness)

            # fitL=np.polyfit(x_left,y_left,1)
            # fit_functionL=np.poly1d(fitL)
            # x1L=120
            # x2L=int(fit_functionL(x1L))
            # y1L=425
            # y2L=int(fit_functionL(x2L))
            # cv2.line(img, (x1L, y1L), (x2L, y2L), color, thickness)


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.

    Returns an image with hough lines drawn.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    draw_lines(line_img, lines)
    return line_img


# Python 3 has support for cool math symbols.

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    `img` is the output of the hough_lines(), An image with lines drawn on it.
    Should be a blank image (all black) with lines drawn on it.

    `initial_img` should be the image before any processing.

    The result image is computed as follows:

    initial_img * α + img * β + λ
    NOTE: initial_img and img must be the same shape!
    """
    return cv2.addWeighted(initial_img, α, img, β, λ)

import os
os.listdir("test_images/")
import imageio
imageio.plugins.ffmpeg.download()


# TODO: Build your pipeline that will draw lane lines on the test_images
# then save them to the test_images directory.

def img_pipeline(img):
    image = np.copy(img)
    # Convert to grayscale
    gray = grayscale(img)

    # apply gaussian smoothing/blurring
    kernel_size = 5
    blur_gray = gaussian_blur(gray, kernel_size)

    # Canny parameters
    low_threshold = 50
    high_threshold = 150
    edges = canny(blur_gray, low_threshold, high_threshold)

    # masking edges of the canny image
    imshape = img.shape
    vertices = np.array([[(110, imshape[0]),  # bottom left
                          (425, 330),  # top left
                          (550, 330),  # top right
                          (875, imshape[0])]],  # bottom right
                        dtype=np.int32)
    masked_edges = region_of_interest(edges, vertices)

    # Define the Hough transform parameters
    # Make a blank the same size as our image to draw on

    rho = 2
    theta = np.pi / 180
    threshold = 5
    min_line_length = 30
    max_line_gap = 20

    # # Run Hough on edge detected image
    # # Output "lines" is an array containing endpoints of detected line segments
    line_image = hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)

    # # Create a "color" binary image to combine with line image
    color_edges = np.dstack((masked_edges, masked_edges, masked_edges))

    # # Draw the lines on the edge image
    lines_edges = weighted_img(img, line_image)

    # Getting color image back


    return lines_edges

#reading all remaining images
solidWhiteCurve= mpimg.imread('test_images/solidWhiteCurve.jpg')
solidYellowCurve= mpimg.imread('test_images/solidYellowCurve.jpg')
solidYellowCurve2= mpimg.imread('test_images/solidYellowCurve2.jpg')
solidYellowLeft= mpimg.imread('test_images/solidYellowLeft.jpg')
whiteCarLaneSwitch= mpimg.imread('test_images/whiteCarLaneSwitch.jpg')

Images=[solidWhiteRight,solidWhiteCurve,solidYellowCurve,solidYellowCurve2,solidYellowLeft,whiteCarLaneSwitch]


#printing out some stats and plotting

print('Solid White Right is:', type(solidWhiteRight), 'with dimensions:', solidWhiteRight.shape)
plt.xticks(np.arange(0,900,50))
plt.imshow(img_pipeline(solidWhiteRight))