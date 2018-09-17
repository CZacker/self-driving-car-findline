# **Finding Lane Lines on the Road** 
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

[![jpg](examples/laneLines_thirdPass.jpg)](https://github.com/CZacker/self-driving-car-findline/blob/master/test_videos_output/solidWhiteRight.mp4)

***
In this project,there are some techniques below:  
1. Color Region  
2. Canny Edges(Gaussian smoothing、Edge Detection)
3. Hough Transform  

with this techniques,then to process video clips to find lane lines in them

***

## Some Test Images

![png](examples/test_image.png)

lane lines are not always the same color, like these test images are in white or yellow. Further, the white lane lines are short or even just some series of dots. So that these lines need to be as one line.

---

## Color Region

first, selecting only yellow and write in the images using the RGB channels

use MATLAB image toolbox color-thresholder,find out color RGB channel region

![png](examples/color-thresholder.png)

```python
def select_rgb_white_yellow(image): 
    converted = convert_hls(image)
    # white color mask
    lower = np.uint8([  0, 200,   0])
    upper = np.uint8([255, 255, 255])
    white_mask = cv2.inRange(converted, lower, upper)
    # yellow color mask
    lower = np.uint8([ 10,   0, 100])
    upper = np.uint8([ 40, 255, 255])
    yellow_mask = cv2.inRange(converted, lower, upper)
    # combine the mask
    mask = cv2.bitwise_or(white_mask, yellow_mask)
    return cv2.bitwise_and(image, image, mask = mask)
```
![png](examples/color-selector.png)

---

It looks pretty good except some other color mix in, so we exclude outside the region of interest by apply a mask.

```python
def region_of_interest(img, vertices):
    #defining a blank mask to start with
    mask = np.zeros_like(img)   
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255       
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)   
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image
```
![png](examples/color-region.png)
