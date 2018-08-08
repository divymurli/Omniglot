import sys
import numpy as np
from scipy.misc import imread
import pickle
import os
import matplotlib.pyplot as plt
import argparse


#Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--path',help='Path to omniglot folder')
parser.add_argument('--save',help='Path to save pickled data',default=os.getcwd())
args = parser.parse_args()
data_path = os.path.join(args.path,"omniglot" ,"python")
train_folder = os.path.join(data_path,'images_background')
val_folder = os.path.join(data_path,'images_evaluation')

#use default save path to save data
save_path = args.save

#For testing purposes
PATH = '/Users/DivyanshuMurli1/Google Drive/Personal_projects/Omniglot/omniglot/python/images_background'


def load_imgs(path,n=0):

	X = []
	y = []

	lang_dict = {}
	curr_y = 0
	#print(curr_y)
	for alphabet in os.listdir(path):
		print("loading alphabet: " + alphabet)
		#print(curr_y)
		lang_dict[alphabet] = [curr_y,None]
		#print(lang_dict)
		alphabet_path = os.path.join(path,alphabet)
		for letter in os.listdir(alphabet_path):
			category_images = []
			#print(letter)
			letter_path = os.path.join(alphabet_path,letter)
			for file_name in os.listdir(letter_path):
				#print(file_name)
				image_path = os.path.join(letter_path,file_name)
				image_to_array = imread(image_path)
				category_images.append(image_to_array)
				y.append(curr_y)
			X.append(np.stack(category_images))
			#except ValueError:
				#print(ValueError)
				#print("error - category_images:", category_images)
			#curr_y +=1

			#print(curr_y)
			lang_dict[alphabet][1] = curr_y 
			curr_y+=1
			#print(curr_y)
	X = np.stack(X)
	y = np.vstack(y)

	return X, lang_dict, y


X_train, c_train, _ = load_imgs(train_folder)
X_val, c_val, _ = load_imgs(val_folder)
#print(X.shape)
#print(c)
#lang_dict indexes the range of each character in the omniglot dataset

#turn on/off to enable/disable 
"""
with open(os.path.join(save_path,"train.pickle"), "wb") as f:
	pickle.dump((X_train,c_train),f)

with open(os.path.join(save_path,"val.pickle"), "wb") as f:
	pickle.dump((X_val,c_val),f)
"""



	



