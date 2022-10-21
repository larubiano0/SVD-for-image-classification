import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
import numpy as np
import time

digits = datasets.load_digits() # MNIST
fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(10, 3))

for ax, image, label in zip(axes, digits.images, digits.target):
    ax[0].imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax[0].set_axis_off()
    u, s, vh = np.linalg.svd(image, full_matrices=True)
    ax[1].imshow(u, cmap=plt.cm.gray_r, interpolation="nearest")
    ax[1].set_axis_off()
    ax[2].imshow(vh, cmap=plt.cm.gray_r, interpolation="nearest")
    ax[2].set_axis_off()
#plt.show()

# flatten the images
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))



### SVD6 rank n matrices ###
def evaluateRank(n):
    dataSVD7 = np.zeros(shape=(n_samples,64))
    clfSVD7 = svm.SVC(gamma=0.0001, C=0.85, kernel='rbf')

    for i,image in enumerate(digits.images):
        u, s, vh = np.linalg.svd(image, full_matrices=True)
        sn = np.diag(s[0:n])
        dataSVD7[i] = (u[:,:n]@sn@vh[:n,:]).reshape(64,)

    X_trainSVD7, X_testSVD7, y_trainSVD7, y_testSVD7 = train_test_split(dataSVD7, digits.target, test_size=0.4, shuffle=False) # El target no cambia
    start = time.time()
    clfSVD7.fit(X_trainSVD7, y_trainSVD7)

    # Predict the value of the digit on the test subset
    predictedSVD7 = clfSVD7.predict(X_testSVD7)
    end = time.time()
    print('Tiempo en segundos de fit y predict para SVD7: ', str(end-start))

    print(f"Classification report for classifier with SVD6 features {clfSVD7}:\n"
        f"{metrics.classification_report(y_testSVD7, predictedSVD7)}\n")

    disp = metrics.ConfusionMatrixDisplay.from_predictions(y_testSVD7, predictedSVD7)
    disp.figure_.suptitle("Confusion Matrix")
    print(f"Confusion matrix:\n{disp.confusion_matrix}")
    plt.show()

evaluateRank(n=3)
evaluateRank(n=5)