import warnings
warnings.filterwarnings(action="ignore")
import matplotlib.pyplot as plt

from sklearn import datasets, svm
digits = datasets.load_digits()

print("digits.target----:", digits.target)
images_and_labels = list(zip(digits.images, digits.target))
print("len(images_and_labels)", len(images_and_labels))

for index, [image,label] in enumerate(images_and_labels[ :5]):
    print("index:", index, "image:\n",image, "label:",label)
    plt.subplot(2,5,index+1)
    plt.axis('on')
    plt.imshow(image,cmap=plt.cm.gray_r,interpolation='nearest')
    plt.title('Training:%i' % label)


n_samples = len(digits.images)
print("n_samples:",n_samples)

imageData = digits.images.reshape((n_samples,-1))
print("After reshaped:len(imageData[0])",len(imageData[0]))

classifier = svm.SVC(gamma=0.001)

classifier.fit(imageData[ :n_samples//2],digits.target[ :n_samples//2])

originalY = digits.target[n_samples //2:]
predictedY= classifier.predict(imageData[n_samples//2: ])

images_and_predictions = list(zip(digits.images[n_samples//2: ],predictedY))

for index, [image,prediction] in enumerate(images_and_predictions[ :5]):
    plt.subplot(2,5,index+6)
    plt.axis('on')
    plt.imshow(image, cmap = plt.cm.gray_r,interpolation='nearest')
    plt.title('Prediction:%i' % prediction)

print("Original Values:", digits.target[n_samples//2: (n_samples//2)+5])
plt.show()


