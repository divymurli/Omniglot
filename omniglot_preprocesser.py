import numpy as np
import pickle
import os
from sklearn.utils import shuffle
import matplotlib.pyplot as plt



#Code adapted from this tutorial https://sorenbouma.github.io/blog/oneshot/

class omniglot_preprocess(object):

	def __init__(self,path,data_subsets):
		self.path = path
		self.data_subsets = data_subsets

	@property
	def load_pickled_data(self):
		data = {}
		classes = {}
		for name in self.data_subsets:
			file_path = os.path.join(self.path,name+'.pickle')
			with open(file_path,'rb') as f:
				(X,c) = pickle.load(f)
				data[name] = X
				classes[name] = c
		return data, classes

	#generate a batch of data so half are same class and half are different. keep batch_size even
	def get_training_batches(self,data,batch_size):
		X = data['train']
		n_classes,n_examples,w,h = X.shape

		#sample batch_size categories to create
		categories = np.random.choice(n_classes,size=(batch_size,),replace=False)
		#print("categories: "+ str(categories))

		#initialize list of arrays for left and right inputs of Siamese network (X)
		pairs = [np.zeros((batch_size,h,w,1)) for i in range(2)]

		#Y
		targets = np.zeros((batch_size,))
		targets[:batch_size//2] = 1

		for i in range(batch_size):
			category = categories[i]
			#print(category)
			#to guarantee that category_2 is different
			category_2 = (categories[i]+np.random.randint(1,n_classes))%n_classes
			#print(category_2)
			idx_1 = np.random.randint(0,n_examples)
			#print("index_1: " + str(idx_1))
			pairs[0][i,:,:,:] = X[category,idx_1,:,:].reshape(w,h,1)
			idx_2 = np.random.randint(0,n_examples)
			#print("index_2: " + str(idx_2))
			
			#same class for first half
			if i < batch_size//2:
				pairs[1][i,:,:,:] = X[category,idx_2,:,:].reshape(w,h,1)

			else:
				pairs[1][i,:,:,:] = X[category_2,idx_2,:,:].reshape(w,h,1)
			#print('next')

		#normalize
		pairs[0] /= 255
		pairs[1] /= 255

		targets = targets.reshape(batch_size,1)

		return pairs,targets

	def create_one_shot_validator(self,data,N):
		#N is the size of the support set
		X = data['val']
		#print("X_shape: " + str(X.shape))
		n_classes_val,n_ex_val,w,h = X.shape

		#sample categories to create
		categories = np.random.choice(n_classes_val,size=(N,),replace=False)
		#print("categories: "+str(categories))
		#benchmark to compare against
		true_category = categories[0]

		true_category = np.asarray([true_category])
		#print("True category: " + str(true_category))

		drawer1,drawer2 = np.random.choice(n_ex_val,(2,),replace=False)
		drawer1 = np.asarray([drawer1])
		#print("drawer1: " + str(drawer1))
		drawer2 = np.asarray([drawer2])
		#print("drawer2: " + str(drawer2))

		support_set_indices = np.asarray(np.random.randint(0,n_ex_val,(N-1,)))
		#print(support_set_indices)
		support_set_indices = np.concatenate((drawer2,support_set_indices),0)
		#print(support_set_indices)
		

		#benchmark image copied N times
		benchmark_image = np.asarray([X[true_category,drawer1,:,:]]*N).reshape(N,w,h,1)
		#print(benchmark_image.shape)
		
		support_set = X[categories,support_set_indices,:,:].reshape(N,w,h,1)
		#dprint((support_set[0] == benchmark_image[0]).all())


		pairs = [benchmark_image,support_set]

		targets = np.zeros((N,))
		targets[0] = 1

		#shuffle along zero axis
		targets, support_set,benchmark_image = shuffle(targets,support_set,benchmark_image)
		targets = targets.reshape(N,1)

		return pairs, targets

	

"""
PATH = '/Users/DivyanshuMurli1/Google Drive/Personal_projects/Omniglot'

Siamese_loader = omniglot_preprocess(PATH,['train','val'])
data, classes = Siamese_loader.load_pickled_data
#print(classes['train'])
print(data['train'].shape)

#pairs,targets = Siamese_loader.get_training_batches(data,32)
#print(pairs[0].shape)


#print(pairs[1].shape)
#print("targets: " + str(targets))


pairs_val, targets_val = Siamese_loader.create_one_shot_validator(data,20)
print(targets_val.shape)
print(pairs_val[0].shape)
"""



