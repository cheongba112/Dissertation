# Requirement of Dataset
- contains of multiple identity and age labels
- each identity has at least two images
- each image has its age label
  
# Data Importing Logic
### get_dataset.py
Firstly, use function to extract each file name and age label of all images in the folder, storing with dictionary(as a hashmap)  
Then append file_name_a, age_label_a, file_name_b into a list according to the dictionary, where file_b has the same identity with file_a  
The get_dataset class open and preprocess file_a and file_b, and turn age_label_a into integer  
loading CACD2000 dataset cost about 3 sec  
  
# Training and Testing Steps
- general cGAN training
- regressor training with pre-trained generator(fixed)
- testing with pre-trained generator and regressor
  


