# Process the pickled dataset into csv format

import cPickle, gzip, numpy

# Load the dataset
f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

# Save dataset as separate csv files
numpy.savetxt("train_images.csv", train_set[0], fmt= "%d", delimiter = ",")
numpy.savetxt("train_labels.csv", train_set[1], fmt= "%d", delimiter = ",")
numpy.savetxt("valid_images.csv", valid_set[0], fmt= "%d", delimiter = ",")
numpy.savetxt("valid_labels.csv", valid_set[1], fmt= "%d", delimiter = ",")
numpy.savetxt("test_images.csv", test_set[0], fmt= "%d", delimiter = ",")
numpy.savetxt("test_labels.csv", test_set[1], fmt= "%d", delimiter = ",")


# Visualize some iamges 
from matplotlib import pyplot as plt

plt.imshow(numpy.reshape(train_set[0][0], (28,-1)), interpolation = 'nearest')
print train_set[1][0]
plt.show()
plt.imshow(numpy.reshape(valid_set[0][0], (28,-1)), interpolation = 'nearest')
print valid_set[1][0]
plt.show()
plt.imshow(numpy.reshape(test_set[0][0], (28,-1)), interpolation = 'nearest')
print test_set[1][0]
plt.show()