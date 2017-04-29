## Traffic Sign Recognition ##

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./examples/image1.png "Visualization"
[image2]: ./examples/image2.png "Original Image"
[image3]: ./examples/image3.png "Normalized Image"
[image4]: ./examples/image4.png "Traffic Sign 1"
[image5]: ./examples/image5.png "Traffic Sign 2"
[image6]: ./examples/image6.png "Traffic Sign 3"
[image7]: ./examples/image7.png "Traffic Sign 4"
[image8]: ./examples/image8.png "Traffic Sign 5"

---
### Files Submitted

- Readme: this file
- [project code](./Traffic_Sign_Classifier.ipynb)

### Dataset Exploration 

#### Dataset Summary
I used the numpy library to calculate summary statistics of the traffic signs data set:

* The size of training set is ? 34,799
* The size of the validation set is ? 4410
* The size of test set is ? 12,630
* The shape of a traffic sign image is ? (32, 32, 3)
* The number of unique classes/labels in the data set is ? 43

#### Exploratory Visulaization
The histogram of the trainig data set shows the distributions of each classes. 

![alt text][image1]


### Design and Test a Model Architecture

#### Preprocessing

As a first step, I decided to convert the images to grayscale because grayscale images can improve accuracy as mentioned in [Yann LeCun's paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf) and can decrease the image dimension by 3 folds. In my case, the gray scale improved validation accuracy from 0.936 to 0.946.

As a last step, I simply normalized the image data using the following method, because optimizer can find the best fit better on well conditioned problem with zero mean and equal variance: (X - 128.) / 128. Here you need to cautious on the value "128." instead of "128". If expression, (X - 128) / 128,  is used, overflowing operation can occur by numpy because X is Unsigned int and cannot handle negative values.

Here is an example of a traffic sign image before and after preprocessing.

- Original image
![alt text][image2]

- Normalized image
![alt text][image3]

#### Model Architecture

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| L1.Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6xW 	|
| L1.RELU					|												|
| L1.Max pooling	      	| 2x2 stride,  outputs 14x14x6xW 				|
| L2.Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16xW 	|
| L2.RELU					|												|
| L2.Max pooling	      	| 2x2 stride,  outputs 5x5x16xW 				|
| L2.Flatten	      	| input 5x5x16xW, output 400xW 				|
| L3.Fully connected		| input 400xW, output 120xW   |
| L3.RELU					|												|
| L3.Dropout	      	| keep_prob 0.6 				|
| L4.Fully connected		| input 120xW, output 84xW   |
| L4.RELU					|												|
| L4.Dropout	      	| keep_prob 0.6 				|
| L5.Fully connected		| input 84xW, output 43   |
| Softmax				| with AdamOptimizer       |

* W is a variable to wide the number of neurons. The final model used "2", which shows better accuracy than "1".
* Dropout is applied on fully connected layer of L3 and L4 only. When applied L1 and L2, the accuracy decreased in my case. 
* Keep_prob for dropout rate used 0.6, which showed better accuracy than any other values "0.7, 0.8, 0.9, 1.0"  


#### Model Training

To train the model, I used the adamoptimizer and following hyperparameters:
  - learning rate : 0.001
  - dropout_prob : 0.6
  - number of epochs : 30
  - batch size : 128
  - initialization sigma : 0.05
  - neuron size : 2 (twice number of neurons in LENET5)

At first, I just tested the optimal parameters without appling image normalization:
  1. I started initialization values with condition no dropout, W=1:
    - sigma = 0.1, test accuracy = 0.859
    - sigma = 0.05, test accuracy = 0.926

  2. Then I applied dropout_prob values with condition, W=1, sigma = 0.05:
    - keep_prob = 0.9 / accuracy train = 0.981, validation = 0.913, test = 0.906
    - keep_prob = 0.8 / accuracy train = 0.993, validation = 0.920, test = 0.910
    - keep_prob = 0.7 / accuracy train = 0.988, validation = 0.908, test = 0.910

  3. Then I applied wider values with condition, W=1, sigma = 0.05, keep_prob = 0.8:
    - wider = 2  / accuracy train = 0.971, validation = 0.900, test = 0.889
    - wider = 3  / accuracy train = 0.989, validation = 0.921, test = 0.912
    - wider = 4  / accuracy train = 0.981, validation = 0.922, test = 0.900
    - wider = 5  / accuracy train = 0.971, validation = 0.901, test = 0.875

As the accuracy wasn't increased to acceptable boundary, I retested after appling image normalization:
  1. I applied learning_rate with condition, W=3, sigma=0.05, keep_prob=0.8
    - learning_rate = 0.001  / accuracy train = 0.997, validation = 0.945, test = 0.939
    - learning_rate = 0.002  / accuracy train = 0.996, validation = 0.946, test = 0.928

  2. I changed overall parameters intuitively: W=2, sigma=0.05, keep_prob=0.6
    - learning_rate = 0.001  / accuracy train = 1.000, validation = 0.957, test = 0.928

Final parameters 
  1. I reduced epochs to avoid overfitting: epochs=6, W=2, sigma=0.05, keep_prob=0.6
    - accuracy train = 0.99, validation = 0.946, test = 0.922
    
Lesson's learned:
  - Need to normalize data first, before tunning parameters, because the paramters won't work in the changed data set. 
  - Need to tuning parameters systemically, because variaty of options exists and manual operations are limited. 
    I'd like to consider [bayseian optimization](https://youtu.be/zhjrfBemz8w) to more efficitvely test.
  - Need to collect more matrix data to evaluate the performance to given model and hyperparameters (loss, weight visulaization) 
  - Need to add data by applying data augmetation to increase accuracy because of biased training data. 
  - Need to create environment to automate changing parameters and finding best fit.  

#### Solution Approach

My final model results were:
* training set accuracy of ? 0.999
* validation set accuracy of ? 0.957
* test set accuracy of ? 0.950

If a well known architecture was chosen:
* What architecture was chosen? LeNET5
* Why did you believe it would be relevant to the traffic sign application? LeNET5 was developed to distingish 10 digits from small size handwrite image(32x32x1) with one color channel, so the input and output data is similar to this problem.  
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well? The training and validation accuracy is similarly increated, and the final test accuracy was also acceptable.  

### Test a Model on New Images

#### Acquiring New Images

Here are five German traffic signs that I found on the web:

![alt text][image4] 

The forth image might be difficult to classify because the speed number is not clear. 

#### Performance on New Images
Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.Speed limit (20km/h)      		| Speed limit (20km/h) 									| 
| 1.Speed limit (30km/h)     			| End of speed limit (80km/h) 										|
| 22.Bumpy road					| Bumpy road											|
| 4.Speed limit (70km/h)	      		| Speed limit (70km/h)					 				|
| 9.No passing			| 34.Turn left ahead      							|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. 

#### Model Certainty - Softmax Probabilities

- First image

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Stop sign   									| 
| 1.55531943e-05     				| 1. Speed limit (30km/h)										|
| 9.65610147e-10					| 32.	End of all speed and passing limits										|
| 1.37152659e-13	      			| 6.	End of speed limit (80km/h)				 				|
| 9.06421748e-14				    | 29. Bicycles crossing     							|

- Second image

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .96         			| 6. End of speed limit (80km/h) 									| 
| .0367     				| 1. Speed limit (20km/h)										|
| .0003					| 5.	Speed limit (80km/h)										|
| .29e-05	      			| 12.		Priority road			 				|
| .24e-06				    | 38. Keep right     							|

- Third image

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| 22. Bumpy road  									| 
| .4e-7     				| 20. 	Dangerous curve to the right										|
| .6e-8					| 15.	No vehicles										|
| .6e-8	      			| 26.	Traffic signals				 				|
| .2e-8				    | 28. Children crossing     							|

- Fourth image

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .64         			| 4  Speed limit (70km/h) 									| 
| .35     				| 25 	Road work									|
| .9e-3					| 26	Traffic signals										|
| .3e-3	      			| 38	Keep right				 				|
| .1e-4				    | 39 	Keep left     							|

- Fifth image

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| 34. Turn left ahead 									| 
| .38e-03     				| 33. Ahead only										|
| .30e-09					| 17.		No entry										|
| .30e-10	      			| 38.			Keep right			 				|
| .12e-10				    | 36.	Go straight or right      							|


