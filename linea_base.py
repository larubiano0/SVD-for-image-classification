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
plt.show()#

# flatten the images
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))

# Create a classifier: a support vector classifier
clf = svm.SVC(gamma=0.0001, C=0.85, kernel='rbf')

# Split data into 60% train and 40% test subsets
X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.4, shuffle=False)
# Learn the digits on the train subset
start = time.time()
clf.fit(X_train, y_train)

# Predict the value of the digit on the test subset
predicted = clf.predict(X_test)
end = time.time()
print('Tiempo en segundos de fit y predict: ', str(end-start))
print(f"Classification report for classifier {clf}:\n"
    f"{metrics.classification_report(y_test, predicted)}\n")

disp = metrics.ConfusionMatrixDisplay.from_predictions(y_test, predicted)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")
plt.show()#

### SVD FEATURES WITH DATA ###

dataSVD = np.zeros(shape=(n_samples,64*3))
clfSVD = svm.SVC(gamma=0.0001, C=0.85, kernel='rbf')

for i,image in enumerate(digits.images):
    u, s, vh = np.linalg.svd(image, full_matrices=True)
    dataSVD[i] = np.concatenate((data[i],u.reshape(64,),vh.reshape(64,)), axis=0)

X_trainSVD, X_testSVD, y_trainSVD, y_testSVD = train_test_split(dataSVD, digits.target, test_size=0.4, shuffle=False) # El target no cambia
start = time.time()
clfSVD.fit(X_trainSVD, y_trainSVD)

# Predict the value of the digit on the test subset
predictedSVD = clfSVD.predict(X_testSVD)
end = time.time()
print('Tiempo en segundos de fit y predict para SVD: ', str(end-start))

print(f"Classification report for classifier with SVD features {clfSVD}:\n"
    f"{metrics.classification_report(y_testSVD, predictedSVD)}\n")

disp = metrics.ConfusionMatrixDisplay.from_predictions(y_testSVD, predictedSVD)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")
plt.show()###

### SVD2 FEATURES DATA REDUCTION ###

dataSVD2 = np.zeros(shape=(n_samples,8*2))
clfSVD2 = svm.SVC(gamma=0.0001, C=0.85, kernel='rbf')

for i,image in enumerate(digits.images):
    u, s, vh = np.linalg.svd(image, full_matrices=True)
    dataSVD2[i] = np.concatenate((u@s,s@vh), axis=0)

X_trainSVD2, X_testSVD2, y_trainSVD2, y_testSVD2 = train_test_split(dataSVD2, digits.target, test_size=0.4, shuffle=False) # El target no cambia
start = time.time()
clfSVD2.fit(X_trainSVD2, y_trainSVD2)

# Predict the value of the digit on the test subset
predictedSVD2 = clfSVD2.predict(X_testSVD2)
end = time.time()
print('Tiempo en segundos de fit y predict para SVD2: ', str(end-start))

print(f"Classification report for classifier with SVD2 features {clfSVD2}:\n"
    f"{metrics.classification_report(y_testSVD2, predictedSVD2)}\n")

disp = metrics.ConfusionMatrixDisplay.from_predictions(y_testSVD2, predictedSVD2)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")
plt.show()##

### SVD3 FEATURES U, VH ONLY ###

dataSVD3 = np.zeros(shape=(n_samples,64*2))
clfSVD3 = svm.SVC(gamma=0.0001, C=0.85, kernel='rbf')

for i,image in enumerate(digits.images):
    u, s, vh = np.linalg.svd(image, full_matrices=True)
    dataSVD3[i] = np.concatenate((u.reshape(64,),vh.reshape(64,)), axis=0)

X_trainSVD3, X_testSVD3, y_trainSVD3, y_testSVD3 = train_test_split(dataSVD3, digits.target, test_size=0.4, shuffle=False) # El target no cambia
start = time.time()
clfSVD3.fit(X_trainSVD3, y_trainSVD3)

# Predict the value of the digit on the test subset
predictedSVD3 = clfSVD3.predict(X_testSVD3)
end = time.time()
print('Tiempo en segundos de fit y predict para SVD3: ', str(end-start))

print(f"Classification report for classifier with SVD3 features {clfSVD3}:\n"
    f"{metrics.classification_report(y_testSVD3, predictedSVD3)}\n")

disp = metrics.ConfusionMatrixDisplay.from_predictions(y_testSVD3, predictedSVD3)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")
plt.show()##

### SVD4 FEATURES U, VH, SIGMA ###

dataSVD4 = np.zeros(shape=(n_samples,64*2+8))
clfSVD4 = svm.SVC(gamma=0.0001, C=0.85, kernel='rbf')

for i,image in enumerate(digits.images):
    u, s, vh = np.linalg.svd(image, full_matrices=True)
    dataSVD4[i] = np.concatenate((s,u.reshape(64,),vh.reshape(64,)), axis=0)

X_trainSVD4, X_testSVD4, y_trainSVD4, y_testSVD4 = train_test_split(dataSVD4, digits.target, test_size=0.4, shuffle=False) # El target no cambia
start = time.time()
clfSVD4.fit(X_trainSVD4, y_trainSVD4)

# Predict the value of the digit on the test subset
predictedSVD4 = clfSVD4.predict(X_testSVD4)
end = time.time()
print('Tiempo en segundos de fit y predict para SVD4: ', str(end-start))

print(f"Classification report for classifier with SVD4 features {clfSVD4}:\n"
    f"{metrics.classification_report(y_testSVD4, predictedSVD4)}\n")

disp = metrics.ConfusionMatrixDisplay.from_predictions(y_testSVD4, predictedSVD4)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")
plt.show()##

### SVD5 FEATURES SIGMA ###

dataSVD5 = np.zeros(shape=(n_samples,8))
clfSVD5 = svm.SVC(gamma=0.0001, C=0.85, kernel='rbf')

for i,image in enumerate(digits.images):
    u, s, vh = np.linalg.svd(image, full_matrices=True)
    dataSVD5[i] = s

X_trainSVD5, X_testSVD5, y_trainSVD5, y_testSVD5 = train_test_split(dataSVD5, digits.target, test_size=0.4, shuffle=False) # El target no cambia
start = time.time()
clfSVD5.fit(X_trainSVD5, y_trainSVD5)

# Predict the value of the digit on the test subset
predictedSVD5 = clfSVD5.predict(X_testSVD5)
end = time.time()
print('Tiempo en segundos de fit y predict para SVD5: ', str(end-start))

print(f"Classification report for classifier with SVD5 features {clfSVD4}:\n"
    f"{metrics.classification_report(y_testSVD5, predictedSVD5)}\n")

disp = metrics.ConfusionMatrixDisplay.from_predictions(y_testSVD5, predictedSVD5)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")
plt.show()##

### SVD6 Encontrado por error ###

dataSVD6 = np.zeros(shape=(n_samples,8*2))
clfSVD6 = svm.SVC(gamma=0.0001, C=0.85, kernel='rbf')

for i,image in enumerate(digits.images):
    u, s, vh = np.linalg.svd(image, full_matrices=True)
    dataSVD6[i] = np.concatenate((u@s,vh@s), axis=0)

X_trainSVD6, X_testSVD6, y_trainSVD6, y_testSVD6 = train_test_split(dataSVD6, digits.target, test_size=0.4, shuffle=False) # El target no cambia
start = time.time()
clfSVD6.fit(X_trainSVD6, y_trainSVD6)

# Predict the value of the digit on the test subset
predictedSVD6 = clfSVD6.predict(X_testSVD6)
end = time.time()
print('Tiempo en segundos de fit y predict para SVD6: ', str(end-start))

print(f"Classification report for classifier with SVD6 features {clfSVD6}:\n"
    f"{metrics.classification_report(y_testSVD6, predictedSVD6)}\n")

disp = metrics.ConfusionMatrixDisplay.from_predictions(y_testSVD6, predictedSVD6)
disp.figure_.suptitle("Confusion Matrix")
print(f"Confusion matrix:\n{disp.confusion_matrix}")
plt.show()
