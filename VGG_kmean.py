import time
import os, os.path
import random
import cv2
import glob
import keras
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
import tensorflow as tf

import pandas as pd
import numpy as np

# dev = string variable; use for input

dev =input("Select device: GPU or CPU ")
if dev == 'CPU' or dev =='cpu':
    print("Selected cpu")
    dev = dev.lower()
    dev = "/"+dev+":0"
    print(dev)

if dev =='GPU' or dev =='gpu':
    print("Selected gpu")
    num = input("Device number(0-2): ")
    dev = dev.upper()
    dev= "/device:"+dev+":"+num
    print(dev)

with tf.device(dev):

	# directory where images are stored
	DIR = "./dataset/eye"

	def dataset_stats():
    
    	# This is an array with the letters available.
    	# If you add another animal later, you will need to structure its images in the same way
    	# and add its letter to this array
    		disease= ['1', '2', '3','4','5','6']
    
    	# dictionary where we will store the stats
    		stats = []
    
    		for eyes in disease:
        		# get a list of subdirectories that start with this character
        		directory_list = sorted(glob.glob("{}/[{}]*".format(DIR, eyes)))
        
        		for sub_directory in directory_list:
            			file_names = [file for file in os.listdir(sub_directory)]
            			file_count = len(file_names)
            			sub_directory_name = os.path.basename(sub_directory)
            			stats.append({ "Code": sub_directory_name[:sub_directory_name.find('-')],
                            			"Image count": file_count, 
                           		"Folder name": os.path.basename(sub_directory),
                            		"File names": file_names})
    
    
    		df = pd.DataFrame(stats)
    
    		return df

	# Show codes with their folder names and image counts
	dataset = dataset_stats().set_index("Code")
	print(dataset[["Folder name", "Image count"]])

# Function returns an array of images whoose filenames start with a given set of characters (PREPROCESSING)
# after resizing them to 224 x 224
#---------------------PREPROCESSING--------------------------
	def load_images(codes):
    
    	# Define empty arrays where we will store our images and labels
    		images = []
    		labels = []
    
    		for code in codes:
        		# get the folder name for this code
        		folder_name = dataset.loc[code]["Folder name"]
        
        		for file in dataset.loc[code]["File names"]:                 
            			# build file path
            			file_path = os.path.join(DIR, folder_name, file)
        
            			# Read the image
            			image = cv2.imread(file_path)

            			# Resize it to 224 x 224
            			image = cv2.resize(image, (224,224))

            			# Convert it from BGR to RGB so we can plot them later (because openCV reads images as BGR)
            			image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            			# Now we add it to our array
            			images.append(image)
            			labels.append(code)

    		return images, labels

	codes = ["1","2","3", "4", "5","6"]
	images, labels = load_images(codes)

	def show_random_images(images, labels, number_of_images_to_show=2):

    		for code in list(set(labels)):

        		indicies = [i for i, label in enumerate(labels) if label == code]
        		random_indicies = [random.choice(indicies) for i in range(number_of_images_to_show)]
        		figure, axis = plt.subplots(1, number_of_images_to_show)

        		print("{} random images for code {}".format(number_of_images_to_show, code))

        		for image in range(number_of_images_to_show):
            			axis[image].imshow(images[random_indicies[image]])
        		plt.show()

	show_random_images(images, labels)

	def normalise_images(images, labels):

    		# Convert to numpy arrays
    		images = np.array(images, dtype=np.float64)
    		labels = np.array(labels)

    		# Normalise the images
    		images /= 255
    
    		return images, labels

	images, labels = normalise_images(images, labels)
	#-------------------------------------------------------------

	#-----get the train and test data-------
	#-----x images data--y labels data------
	def shuffle_data(images, labels):
		# Set aside the testing data. We won't touch these until the very end.
    		X_train, X_test, y_train, y_test = train_test_split(images, labels)#input dari preprocessing normalization
    
    		return X_train, y_train

	X_train, y_train = shuffle_data(images, labels)
	#----------------------------------------

	# Load the models with ImageNet weights
	vgg16_model = keras.applications.vgg16.VGG16(include_top=False, weights="imagenet", input_shape=(224,224,3)) #taken from github vgg16 architecture

	def covnet_transform(covnet_model, raw_images): #covnet is CNN model

    		# Pass our training data through the network
    		pred = covnet_model.predict(raw_images) #validate data dalam vgg16 (dapat last matrix dr CNN vgg)

    		# Flatten the array
    		flat = pred.reshape(raw_images.shape[0], -1) #kira feature dia guna flatten
    
    		return flat

	vgg16_output = covnet_transform(vgg16_model, X_train)

	print("VGG16 flattened output has {} features".format(vgg16_output.shape[1]))#got after flatten vgg16_output size last lepas process dia
	#-----------------------------------------------------------------------------------
	#--------------------PCA------------------------------------------------------------
	# Function that creates a PCA instance, fits it to the data and returns the instance
	def create_fit_PCA(data, n_components=None):
    
    		p = PCA(n_components=n_components, random_state=728)
    		p.fit(data) #train dalam pca
    
    		return p

	# Create PCA instances for each covnet output
	vgg16_pca = create_fit_PCA(vgg16_output) #dari output vgg16 train guna pca

	# Function to plot the cumulative explained variance of PCA components	
	# This will help us decide how many components we should reduce our features to
	def pca_cumsum_plot(pca):
    		plt.plot(np.cumsum(pca.explained_variance_ratio_))
    		plt.xlabel('number of components')
    		plt.ylabel('cumulative explained variance')
    		plt.show()

	# Plot the cumulative explained variance for each covnet
	pca_cumsum_plot(vgg16_pca) #plot PCA punya 

	# PCA transformations of covnet outputs
	vgg16_output_pca = vgg16_pca.transform(vgg16_output)

	#-----------------------------------------------------------------------------------
	#-------------kmeantimefunction&train-----------------------------------------------------------------
	def create_train_kmeans(data, number_of_clusters=len(codes)):
    		# n_jobs is set to -1 to use all available CPU cores. This makes a big difference on an 8-core CPU
    		# especially when the data size gets much bigger. #perfMatters
    
    		k = KMeans(n_clusters=number_of_clusters, n_jobs=-1, random_state=728)

    		# Let's do some timings to see how long it takes to train.
    		start = time.time()

    		# Train it up
    		k.fit(data)

    		# Stop the timing 
    		end = time.time()
    
    		# And see how long that took
    		print("Training took {} seconds".format(end-start))
    
    		return k
	#-----------------------------------------------------------------------------------

	# Let's pass the data into the algorithm and predict who lies in which cluster. 
	# Since we're using the same data that we trained it on, this should give us the training results.

	# Here we create and fit(train) a KMeans model with the PCA outputs
	print("**********************************TIME****************************************")
	print("***********************************PCA****************************************")
	print("KMeans (PCA): \n")
	print("VGG16")
	K_vgg16_pca = create_train_kmeans(vgg16_output_pca)
	print("*****************************************************************************")
	# Let's also create models for the covnet outputs without PCA for comparison
	print("********************************withoutPCA************************************")
	print("KMeans: \n")
	print("VGG16:")
	K_vgg16 = create_train_kmeans(vgg16_output)
	print("*****************************************************************************")
	print("*****************************************************************************\n")

	# Now we get the custer model predictions
	# KMeans with PCA outputs
	k_vgg16_pred_pca = K_vgg16_pca.predict(vgg16_output_pca) #validate guna reduction PCA
	# KMeans with CovNet outputs
	k_vgg16_pred = K_vgg16.predict(vgg16_output) #validate guna cluster 
	print(vgg16_output)
	#----------------------------------------------------------------------------------------
	def cluster_label_count(clusters, labels):
    
    		count = {}
    
    		# Get unique clusters and labels
    		unique_clusters = list(set(clusters))
    		unique_labels = list(set(labels))
    
    		# Create counter for each cluster/label combination and set it to 0
    		for cluster in unique_clusters:
        		count[cluster] = {}
        
        		for label in unique_labels:
           		 count[cluster][label] = 0
    
    	# Let's count
    		for i in range(len(clusters)):
        		count[clusters[i]][labels[i]] +=1
    
    		cluster_df = pd.DataFrame(count)
    
    		return cluster_df

	# Cluster counting for VGG16 Means
	vgg16_cluster_count = cluster_label_count(k_vgg16_pred, y_train) #vggkmeandata as x and y as label
	vgg16_cluster_count_pca = cluster_label_count(k_vgg16_pred_pca, y_train)

	print("********************clustering*******************************")
	print("KMeans VGG16: ")
	print(vgg16_cluster_count) #C=19 D=18 number dia random semua 37 utk validate dari 50 gmbr
	print("KMeans VGG16 (PCA): ")
	print(vgg16_cluster_count_pca) #C=19 D=18
	print("*************************************************************\n")

	print("***************************untuk manual adjust**********************************")
	# Manually adjust these lists so that the index of each label reflects which cluter it lies in
	vgg16_cluster_code = ["1","2","3", "4","5","6"]
	vgg16_cluster_code_pca = ["1","2","3","4", "5","6"]

	vgg16_pred_codes = [vgg16_cluster_code[x] for x in k_vgg16_pred]
	vgg16_pred_codes_pca = [vgg16_cluster_code_pca[x] for x in k_vgg16_pred_pca]

	from sklearn.metrics import accuracy_score, f1_score

	def print_scores(true, pred):
    		acc = accuracy_score(true, pred)
    		f1 = f1_score(true, pred, average="macro")
    		return "\n\tF1 Score: {0:0.8f}   |   Accuracy: {0:0.8f}".format(f1,acc)

	print("KMeans VGG16:", print_scores(y_train, vgg16_pred_codes))
	print("KMeans VGG16 (PCA)", print_scores(y_train, vgg16_pred_codes_pca))

