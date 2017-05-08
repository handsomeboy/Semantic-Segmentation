## Welcome to Our Project Pages

Chen Wang, Xiahe Liu

### Introduction

Our project is about exploring the topic of semantic segmentation. We investigated into fully convolutional neural networks, which is the state-of-the-art techniques to solve this problem. What is more, we went one step further to make some applications based on what we have found.

### Traditional Image Segmentation
During CS 766 lectures, we have learned several traditional image segmentation methods, like K-means and mean-shift. Most of these methods are based on clustering algorithms. We use spatial and RGB information for each pixel and cluster pixels with similar characteristics into the same cluster. They could have good result in some situations, however, the major drawback of these traditional image segmentation is that the clustering algorithms don't know what each cluster represents. In other words, it loses the semantic information. Objects of the same class may be clustered to different clusters because of discrepancies in appearance or location.

We implemented the traditional K-means and mean-shift algorithms in Matlab. Here are some of the results we got:
![alt text](/fig/kmeans.jpg)
![alt text](/fig/meanshift.jpg)

### Semantic Segmentation
Then what is semantic segmentation? How is it different from traditional image segmentation? Semantic segmentation is the task of clustering parts of images together which belong to the same object class. In other words, it is doing image classification for every pixel in the image.

Here are some of the examples of semantic segmentation:
![alt text](/fig/exp1l.jpg)
![alt text](/fig/exp1r.jpg)
In this example, all persons are labeled pink and all bicycles are green.

![alt text](/fig/exp1l.jpg)
![alt text](/fig/exp1r.jpg)
In this example, the desk part is yellow and all chairs are red.

### Fully Convolutional Neural Network

![alt text](/fig/fcn.png)
Our project used fully convolutional neural (FCN) network to realize semantic segmentation. This neural network structure is proposed in [Fully convolutional networks for semantic segmentation](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf). The input of FCN is an image of arbitrary size, with RGB information for each pixel. The output is an indexed image of the same size. Each pixel represents a class it is clustered to. Thus, it completes a Pixelwise end-to-end image classification.

![alt text](/fig/cnn.png)
For traditional convolutional neural network which focus on image classification task, like AlexNet, GoogLeNet, VGG net, etc., the whole network stops after several fully connected layers following some convolutional layers. The last fully connected layer will output the score for each category and classify the image to be the category with the highest score.

 ![alt text](/fig/unet.png)
This is one of the typical structures of FCN, which is from [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597.pdf). There are two parts in this structure. The first part is the contracting path, which is actually from the traditional image classification convolutional neural network structure. It will downsample the image and extract the features during training. The other part is the expanding path, which is the reverse process of the first part, upsampling the dense features back into full size images. By first contracting and then expanding, we get an output which has the same size with the input image, with a label for each pixel in the image.

### Implementation and Experiments
We used deep learning framework [caffe](http://caffe.berkeleyvision.org/) to build up the training model. We deployed it on Google Cloud, with Ubuntu 16.04 operating system. The computing engine we were using has 16 cores, 60G memory and 500G disk. We used [Pascal VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) Segmentation contest dataset, which has 1464 images in training set, 1449 images in validation set and 21 classes in total.

![alt text](/fig/fcn8s1.jpg)
![alt text](/fig/fcn8s2.jpg)
![alt text](/fig/fcn8s3.jpg)
![alt text](/fig/fcn8s4.jpg)
![alt text](/fig/fcn8s5.jpg)
![alt text](/fig/fcn8s6.jpg)
Here is the structure we finally used for training. It is pretty complicated, with a series of convolutional, relu and pooling layers at first, followed by some fully connected layers and finally do some deconvolutional(upsampling) layers.

As you could imagine, this huge network will take forever if we train it from scratch. Also, our data set is very limited to get good results. Thus, we decided to fine-tune it from VGG net. VGG net is a pre-trained image classification model from ImageNet dataset. We loaded weights from this model to our fully convolutional neural network, fine-tuning all layers by back-propagation through the whole net. Also, we combine coarse, high layer information with fine, low layer information to get more accurate results.

Here are some results:
![alt text](/fig/result1l.jpg)
![alt text](/fig/result1r.jpg)
![alt text](/fig/result2l.jpg)
![alt text](/fig/result2r.jpg)

### An Advertising Application Based on Semantic Segmentation
We implemented an automatic advertising application based on the results of the semantic segmentation, the user can upload the original image and advertisemnet, then CNN is used to get the area that can be used to place advertisemnet (in our case, it is the area of TV/monitors). After that we will finding more accurate boundry of the target area, transform the advertisement and warp it to origianl image.

#### An Overview of the Application

![alt text](/fig/app.png)

Here is an overiew of our automatic advertising application, it contains four modules, UI, converlutional neural network (CNN), boundary finding and ads warp. Start from the UI, the user can upload the original image and ads, then UI will send the image to CNN and ads to ads warp module. With pre-trained model, CNN can do pixel-wise semantic segementation over the input image and the outout is a labeled image. The with the result of semantic segementation, the boundary finding module tries to find more accurate and regular area of TV/monitors if exists. The boundary finding module will output corner points of one of the largest target area to ads warp module. Finally, the advertisment will be transformed accoding to these corner points and is warpped to the target area of the origimal image.

#### UI Design

![alt text](/fig/ui.png)

To be consistent with the CNN part, we use python to complete all the other modules. We choosed [PyQt](https://wiki.python.org/moin/PyQt), which is a cross-platform GUI framework to implement the UI module of this application. As we can see from the screen shot of the UI, we can use it to upload image/ads and do advertising. All the input and output will be displayed in fixed area, and all the images will be resized to fit the given area and do all the following computation and transformation. Note that the size of the original/output image is changed to 480*640, and the size of the advertisement is 240*320. Also resize the images with large size can accelerate the computation of semantic segmentation (CNN). When the buttons are pressed, the response function will trigger corresponding actions, for example, we the "advertising" button is pressed, the input image will be transferred to CNN and do semantic segmentation, then do boundary finding and ads warp, after that the final result will be diaplayed at the right image window.

#### Boundary Detection
However the result form semantic segmentation is pixel-wised, it labels each pixel and the shape of the target area is not a regular one, but it cannot be directly used to do ads warp. The goal for this module is to find a regular target area according the results of semantic segmentation. And for this section, we use the one of the image processing library [scikit-image](http://scikit-image.org/), which contains good implementation of some boundary finding algorithms. We have tried several methods that will be describled as follows,

1. Approximate and subdivide polygons
For this method, it cooperate [Douglas–Peucker algorithm](https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm) and [B-spline](https://en.wikipedia.org/wiki/B-spline). The input is a set of points in given order and an acceptable distance between approximated line and the input points, the algorithm will approximate and divide the lines iteratively until the output satisfies the given distance threshold. Note that, larger threshold will result in more regular polygons, however, the threshold could be different for different input. The output is shown as follows,

![alt text](/fig/boundary1.png)

However, this method cannot guarantee find regular target area for different input and threshold, and it is hard to adjust the threshold for all images, thus we will explore other ways to get the boundary of the target area.

2. Active contour
Another way is to use the [active contour model](https://en.wikipedia.org/wiki/Active_contour_model), but this model reuqires an intial contour as input. We tried two ways to get the initial contour, the first is to use [Marching Squares](https://en.wikipedia.org/wiki/Marching_squares) method, which can find a initial contour, as we can see in the following result, the read line represents the initial contour while the blue line is the result of active contour algorithm, and the performance of this approach is not satisfying.

![alt text](/fig/boundary2.png)

Then we try the second way to get an initial contour, which is to do canny edge detection over the result of semantic segementation, and the find the largest connected components, which reperesents the largest detected area. Then we get the corner point of that area and draw a corresponding rectangle as the initial contour (red line), the output can become better as the blue line shows.

![alt text](/fig/boundary3.png)

To sum up, we found that the output boundary of the area is related to the input boundary. The methods metioned above cannot give desired boundary of the target. To simplify the problem, we assume that the area of TV/monitors can be considered as a polygon, we choose the init contour of the second strategy of active contour finding as our target area, and we can see the results after warping is acceptable. But this part can still be improved.

#### Advertisment Warpping
The advertisement warping can be done with the following steps,
1. Get the corner points from boundary detection.
2. Calculate transformation from Ads to target area of image
3. Weighted mask of image and Ads
4. Merge Image and Ads

Note that, the weighted masks are generated using the distance between the nearest target/background pixel. As the following example shows, a) is the original image (binary), b) are the distance from each pixel to its nearest pixel that has value 1. Then we set a threshold to modifiy all the distance larget than th to th, and normalize all the distances to get the weight mask. In this example, we choose th=1.4, then the center pixel in b) is set to 1.4, and then all the pixels are normalized to get c).

![alt text](/fig/mask.png)

The following figure is one of the examples.

![alt text](/fig/output.jpg)

#### Demo
[Here is demo of our automatic advertising app](/files/demo2x.mp4)

### Other Resources

#### Project Proposal
[Here is our project proposal](https://github.com/shynehua/Semantic-Segmentation/edit/master/files/proposal.pdf)

#### Final Project Presentations
[Here is our slides for presetation](https://docs.google.com/presentation/d/1a2Luw1wK1LthmIWZab4sfcmiJ3vgsgiugX3XXOmfmAE/edit?usp=sharing)

#### References
- [Paper 1, CVPR 2016](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf)
- [Paper 2, CVPR 2015](https://arxiv.org/pdf/1504.01013.pdf)
