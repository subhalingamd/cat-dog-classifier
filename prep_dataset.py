from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from os import listdir


# Folder containing training examples
folder='train/'

photos,labels = list(),list()

i=1

for file in listdir(folder):

	print('Loading image',i,end='...')

	# determine class
	output=0
	if file.startswith('dog'):
		output=1

	# load image
	photo = load_img(folder+file, target_size=(50,50))

	# convert to array
	photo = img_to_array(photo)

	#append to list
	photos.append(photo)
	labels.append(output)

	print('Done\n')
	i+=1


print('Converting to array',end='...')

# convert list to array
photos = np.asarray(photos)
labels = np.asarray(labels)

print('Done\nSaving...')

# store dataset
np.save('dataset-photos.npy',photos)
np.save('dataset-labels.npy',labels)


