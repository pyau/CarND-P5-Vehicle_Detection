##Vehicle Detection Project

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/image0077.png
[image1a]: ./output_images/image14.png
[image2]: ./output_images/img_and_hog.png
[image3]: ./examples/sliding_windows.jpg
[image4]: ./output_images/labels.png
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in in lines 18 through 38 of the file called `vehicle_detection.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]
![alt text][image1a]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.  The code is in lines 40-51 in `vehicle_detection.py`.

Here is an example using the `RGB` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)`, `cells_per_block=(2, 2)` and `hog_channel=0`:


![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and found out the best parameters to use are `color_space="YCrCb"`, `orient=9`, `pixels_per_cell=(8,8)`, `cells_per_block=(2,2)`, `hog_channel=ALL`.  The test accuracy is above 99.2% when using linear SVC to train classifier.

I found both HSV and YCrCb yield good test accuracy during training.  For this project, using one channel for hog_channel is only good for visualization.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using spatial features, color histogram features and hog features. I extract these features in lines 66-103 in `vehicle_detection.py`. First, I convert the image to the desired color space (YCrCb), then get different features from the images.

The training code is in line 277-317.  Before training, I use a StandardScaler to remove mean and scale to unit variance (line 298-299). Then I apply linear svc to fit the data. 90% of data is used for training, and 10% of data for testing.

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The code for sliding window is in `find_cars()` function, lines 457- 516 in `vehicle_detection.py`.  With sliding window, Hog feature at the same pixel region will be calculated multiple times, especially with overlaps.  The method I use is to first calculate hog features for the full image, and then subsample the hog feature for each region of interest. The window size that I use is 64x64, and is divided into 8x8 cells.  Then I define each step the subsampling would move forward 2 cells.  That means my sliding window searches have 75% overlap.

I used multiple scales to search only for videos, in lines 566-569 in `vehicle_detection.py`.  I call the `find_cars()` function with a scale variable.  The scale variable is first used to scale the image, before apply a fixed size sliding window search of size 64.  This has similar effect as using different sizes of search window.  Apply scales 1.0 and 2.0 yields good result, because cars from both near the camera and far from camera can be detected.

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./result.mp4)

Or this can be watched in youtube link below:

<a href="http://www.youtube.com/watch?feature=player_embedded&v=aG9QYPUj8So
" target="_blank"><img src="https://github.com/pyau/CarND-P5-Vehicle_Detection/blob/master/video_screenshot.png?raw=true" 
alt="Click to watch video" width="480" height="360" border="10" /></a>

####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

Lines 555-563 show my solution for filtering out false positives.  I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap.  I combined the heatmap for the last 30 frames of the video, so I get a greater confidence level.  Then I apply thresholding (lines 519-521) to the heatmap to take out false positives.

Then I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  


---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

In around 41st second of the video with more trees and shadows, there are a lot of false positives if I don't use the filtering that I described above.  The result video does not have any false positive present, after tuning thresholding and other parameters for a while.

The white car in the video was not detected for a short period of time, when it was farther away from the camera.  In general, the detection of the white car is not very robust.  I think more positive training data is required to make the detection more robust in this case.

I also combined result from the advanced lane finding project with vehicle detection.  The result video can be seen below.  My bounding boxes around vehicle is not very tight.  Further tuning of the threshold parameter, and filtering parameters should improve the performance.

<a href="http://www.youtube.com/watch?feature=player_embedded&v=EaG2X9x1B9w
" target="_blank">Link to combined result video</a>