## Welcome to Our Project Pages

Chen Wang, Xiahe Liu

### Introduction

### Traditional Image Segmentation

### Semantic Segmentation

### An Advertising Application Based on Semantic Segmentation
We implemented an automatic advertising application based on the results of the semantic segmentation, the user can upload the original image and advertisemnet, then CNN is used to get the area that can be used to place advertisemnet (in our case, it is the area of TV/monitors). After that we will finding more accurate boundry of the target area, transform the advertisement and warp it to origianl image.

#### An Overview of the Application
![alt text](https://github.com/shynehua/Semantic-Segmentation/edit/master/fig/app.png)
Here is an overiew of our automatic advertising application, it contains four modules, UI, converlutional neural network (CNN), boundary finding and ads warp. Start from the UI, the user can upload the original image and ads, then UI will send the image to CNN and ads to ads warp module. With pre-trained model, CNN can do pixel-wise semantic segementation over the input image and the outout is a labeled image. The with the result of semantic segementation, the boundary finding module tries to find more accurate and regular area of TV/monitors if exists. The boundary finding module will output corner points of one of the largest target area to ads warp module. Finally, the advertisment will be transformed accoding to these corner points and is warpped to the target area of the origimal image.

#### UI Design
![alt text](https://github.com/shynehua/Semantic-Segmentation/edit/master/fig/ui.png)
To be consistent with the CNN part, we use python to complete all the other modules. We choosed [PyQt](https://wiki.python.org/moin/PyQt), which is a cross-platform GUI framework to implement the UI module of this application. As we can see from the screen shot of the UI, we can use it to upload image/ads and do advertising. All the input and output will be displayed in fixed area, and all the images will be resized to fit the given area and do all the following computation and transformation. Note that the size of the original/output image is changed to 480*640, and the size of the advertisement is 240*320. Also resize the images with large size can accelerate the computation of semantic segmentation (CNN). When the buttons are pressed, the response function will trigger corresponding actions, for example, we the "advertising" button is pressed, the input image will be transferred to CNN and do semantic segmentation, then do boundary finding and ads warp, after that the final result will be diaplayed at the right image window.

#### Boundary Detection
However the result form semantic segmentation is pixel-wised, it labels each pixel and the shape of the target area is not a regular one, but it cannot be directly used to do ads warp. The goal for this module is to find a regular target area according the results of semantic segmentation. And for this section, we use the one of the image processing library [scikit-image](http://scikit-image.org/), which contains good implementation of some boundary finding algorithms. We have tried several methods that will be describled as follows,

1. Approximate and subdivide polygons 
For this method, it cooperate [Douglas–Peucker algorithm](https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm) and [B-spline](https://en.wikipedia.org/wiki/B-spline). The input is a set of points in given order and an acceptable distance between approximated line and the input points, the algorithm will approximate and divide the lines iteratively until the output satisfies the given distance threshold. Note that, larger threshold will result in more regular polygons, however, the threshold could be different for different input. The output is shown as follows, 
![alt text](https://github.com/shynehua/Semantic-Segmentation/edit/master/fig/boundary1.png)
However, this method cannot guarantee find regular target area for different input and threshold, and it is hard to adjust the threshold for all images, thus we will explore other ways to get the boundary of the target area.

2. Active contour
Another way is to use the [active contour model](https://en.wikipedia.org/wiki/Active_contour_model), but this model reuqires an intial contour as input. We tried two ways to get the initial contour, the first is to use [Marching Squares](https://en.wikipedia.org/wiki/Marching_squares) method, which can find a initial contour, as we can see in the following result, the read line represents the initial contour while the blue line is the result of active contour algorithm, and the performance of this approach is not satisfying.
![alt text](https://github.com/shynehua/Semantic-Segmentation/edit/master/fig/boundary2.png)

Then we try the second way to get an initial contour, which is to do canny edge detection over the result of semantic segementation, and the find the largest connected components, which reperesents the largest detected area. Then we get the corner point of that area and draw a corresponding rectangle as the initial contour (red line), the output can become better as the blue line shows.
![alt text](https://github.com/shynehua/Semantic-Segmentation/edit/master/fig/boundary3.png)

To sum up, we found that the output boundary of the area is related to the input boundary. The methods metioned above cannot give desired boundary of the target. To simplify the problem, we assume that the area of TV/monitors can be considered as a polygon, we choose the init contour of the second strategy of active contour finding as our target area, and we can see the results after warping is acceptable. But this part can still be improved.

#### Advertisment Warpping
The advertisement warping can be done with the following steps,
1. Get the corner points from boundary detection.
2. Calculate transformation from Ads to target area of image
3. Weighted mask of image and Ads
4. Merge Image and Ads

Note that, the weighted masks are generated using the distance between the nearest target/background pixel. As the following example shows, a) is the original image (binary), b) are the distance from each pixel to its nearest pixel that has value 1. Then we set a threshold to modifiy all the distance larget than th to th, and normalize all the distances to get the weight mask. In this example, we choose th=1.4, then the center pixel in b) is set to 1.4, and then all the pixels are normalized to get c).
![alt text](https://github.com/shynehua/Semantic-Segmentation/edit/master/fig/mask.png)

The following figure is one of the examples.
![alt text](https://github.com/shynehua/Semantic-Segmentation/blob/master/fig/output.jpg)

#### Demo
[Here is demo of our automatic advertising app](https://github.com/shynehua/Semantic-Segmentation/tree/master/files/demo2x.mp4)

### Other Resources

#### Project Proposal
[Here is our project proposal](https://github.com/shynehua/Semantic-Segmentation/edit/master/files/proposal.pdf)

#### Final Project Presentations
[Here is our slides for presetation](https://docs.google.com/presentation/d/1a2Luw1wK1LthmIWZab4sfcmiJ3vgsgiugX3XXOmfmAE/edit?usp=sharing)

#### References
- [Paper 1, CVPR 2016](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf)
- [Paper 2, CVPR 2015](https://arxiv.org/pdf/1504.01013.pdf)
