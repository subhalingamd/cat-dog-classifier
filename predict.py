from keras.preprocessing.image import load_img,img_to_array
import numpy as np
from keras.models import load_model

# Command line input of file name in 'test' directory (with extension)
name=input('Enter file name in test/ directory:\t')
folder='test/'

# Load image and reshape
img=load_img(folder+name,target_size=(50,50))
x=np.zeros((1,50,50,3))
x[0,...]=img_to_array(img)


# Load model
model=load_model('model.h5')


y=model.predict(x)

# print(y) : <0.5=> Cat;  else Dog

if np.round(y)==0:
	print("Its a cat!")
else:
	print("Its a dog!") 