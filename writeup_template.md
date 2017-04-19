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
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
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
* The size of the validation set is ? 
* The size of test set is ? 12,630
* The shape of a traffic sign image is ? (32, 32, 3)
* The number of unique classes/labels in the data set is ? 43

#### Exploratory Visulaization
The histogram of the trainig data set shows the distributions of each classes. 

![alt text][image1]


### Design and Test a Model Architecture

#### Preprocessing

At first, I only applied the essential data preprocessing, normalization. I applied a quick way to approximately normalize the image data by using (pixel - 128.)/128, instead of using mean zero and equal variance. 

Here is an example of a traffic sign image before and after normalization.

![alt text][image2]



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

#### Solution Approach

My final model results were:
* training set accuracy of ?
* validation set accuracy of ? 
* test set accuracy of ? 

If a well known architecture was chosen:
* What architecture was chosen? LeNET5
* Why did you believe it would be relevant to the traffic sign application? LeNET5 was developed to distingish 10 digits from small size handwrite image(32x32x1) with one color channel, so the input and output data is similar to this problem.  
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well? The training and validation accuracy is similarly increated, and the final test accuracy was also acceptable.  

###Test a Model on New Images

#### Acquiring New Images

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The forth image might be difficult to classify because the speed number is not clear. 

#### Performance on New Images
Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 0.Speed limit (20km/h)      		| Speed limit (20km/h) 									| 
| 1.Speed limit (30km/h)     			| Speed limit (30km/h) 										|
| 22.Bumpy road					| Bumpy road											|
| 4.Speed limit (70km/h)	      		| Speed limit (70km/h)					 				|
| 9.No passing			| 34.Turn left ahead      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. 

#### Model Certainty - Softmax Probabilities
- First image
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|

- Second image
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|

- Third image
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|

- Fourth image
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|

- Fifth image
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


