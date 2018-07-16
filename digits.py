#Handwritten digit classifier using MNIST datasets
#written by @_Rakesh_pandey
#Importing modules
from scipy import misc
import pandas as pd
from sklearn import neural_network

#For making image suitable for classifier
def image_processor(image_name):
   # Modifying self mde image for testing
   img = misc.imread(image_name)
   img = misc.imresize(img, (28,28))
   img = img.astype('int64')
   #x_test will contain the pixels of image
   x_test = []
   for eachRow in img:
      for eachPixel in eachRow:
         x_test.append(sum(eachPixel)/3.0)
   return x_test

#Setting up classifier
train=pd.read_csv("mnist_train.csv").values
features=train[:20000,1:]
labels=train[:20000,0]
clf = neural_network.MLPClassifier(max_iter=10000,random_state=1) #Neural network classifier
clf.fit(features, labels)

# For testing own image
#Change image name eg . from predict.jpg to predict2.jpg
#image predict is of 50x50 pixels and image predict2 is of 40x40 pixels
prediction=image_processor("predict.jpg")
print("The predicted value of image is :")
print(clf.predict([prediction]))
