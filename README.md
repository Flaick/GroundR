# GroundR
Pytorch implementation of the paper \<Grounding of Textual Phrases in Images by Reconstruction\>      
<img src="https://github.com/Flaick/GroundR/blob/master/src/Fig1.jpg" width="900" height="500">             

# Reference: 
https://arxiv.org/pdf/1511.03745.pdf         
https://github.com/acambray/GroundeR-PyTorch

# Data Preparation
The dataset used is Flickr 30k, download link: https://github.com/BryanPlummer/flickr30k_entities         
You can find three different types of data:       
1. Images (.png): All training images
2. Sentences (.txt): Each .txt file corresponds to one image. There are multiple sentences in one .txt file which refer to different object, respectively         
3. Annotations (.xml): One .xml file corresponds to one image, which refer to bounding boxes for different objects
Store them into different folders under the folder "data"


## Step 1       
Run the process_dataset.py              
The result will be stored into ./visualfeatures_data , ./annotation_data, ./id2idx_data automatically.
## Step 2     
Run the train_unsupervised_withval.py        
## Result
<img src="https://github.com/Flaick/GroundR/blob/master/learning_profile.png" width="600px" height="500px">             
<img src="https://github.com/Flaick/GroundR/blob/master/accuracies.png" width="600px" height="500px">         

## Demo        
In progress


