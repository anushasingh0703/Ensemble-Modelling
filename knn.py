#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install numpy')
get_ipython().system('pip install scikit-learn')
get_ipython().system('pip install matplotlib')


# In[3]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os
from PIL import Image
import matplotlib.pyplot as plt
from skimage import io, color, feature, transform
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
import pickle



# In[9]:


# Loading the dataset
SkinCancer_dir = "C:/Anusha/PBL 2/Dataset/diseases"


# In[15]:


# training
# Initialize empty lists for feature vectors and labels
X = []
y = []
desired_size = (128, 128)
i = 0

# Loop through subdirectories, each representing a class or label
for label in os.listdir(SkinCancer_dir):
    label_dir = os.path.join(SkinCancer_dir, label)
    i += 1
    print(i)
    # if i == 2:
    #     break

    # Loop through image files in each label directory
    for filename in os.listdir(label_dir):
        if filename.endswith((".jpg")):  # Adjust file extensions as needed
            image_path = os.path.join(label_dir, filename)

            # Load the image and convert it to grayscale
            img = io.imread(image_path)
            gray_img = color.rgb2gray(img)
            resized_img = transform.resize(gray_img, desired_size)
            # Extract features from the image (e.g., Histogram of Oriented Gradients - HOG)
            feature_vector = feature.hog(resized_img, pixels_per_cell=(8, 8))

            # Append the feature vector and label to the lists
            X.append(feature_vector)
            y.append(label)

# Convert lists to NumPy arrays
X = np.array(X)
y = np.array(y)
print(X)
print(y)
# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a KNN classifier with a specified number of neighbors
knn = KNeighborsClassifier(n_neighbors=3)  # Adjust 'n_neighbors' as needed

# Fit the classifier to the training data
knn.fit(X_train, y_train)

# Assuming 'knn' is your trained KNN model
#with open('C:/Users/vijay/Desktop/pythonProject3/KNN/knn3.pkl', 'wb') as file:
    #pickle.dump(knn, file)
    #print("Model dumped at - C:/Users/vijay/Desktop/pythonProject3/KNN/knn3.pkl")

# Make predictions on the test data
y_pred = knn.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)



# In[14]:


import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os
from PIL import Image
import matplotlib.pyplot as plt
from skimage import io, color, feature, transform
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
import pickle

# Loading the Validation dir
SkinCancer_VAL_dir = "C:/Anusha/PBL 2/Dataset/VAL1"
desired_size = (128, 128)

# Load the KNN model from the pickle file
#with open('C:/Users/vijay/Desktop/pythonProject3/KNN/knn5.pkl', 'rb') as file:
#knn = pickle.load(file)

# Loop through image files in each label directory
for filename in os.listdir(SkinCancer_VAL_dir):
    if filename.endswith((".jpg")):  # Adjust file extensions as needed
        image_path = os.path.join(SkinCancer_VAL_dir, filename)

        # Load the image and convert it to grayscale
        img = io.imread(image_path)
        # Preprocess the input image (convert to grayscale and extract features, e.g., HOG)
        gray_image = color.rgb2gray(img)
        resized_img = transform.resize(gray_image, desired_size)
        feature_vector = feature.hog(resized_img, pixels_per_cell=(8, 8))

        # Reshape the feature vector to match the training data (if necessary)
        feature_vector = feature_vector.reshape(1, -1)

        # Make a prediction using the KNN model
        predicted_label = knn.predict(feature_vector)

        # Display the predicted label
        print("filename | Predicted Label:", filename, predicted_label[0])

# # validation
# # Load the single input image
# image_path = 'C:/Users/vijay/Desktop/pythonProject3/KNN/datasetSkin/Test/AK/1.jpg'
# input_image = io.imread(image_path)
#
# # Preprocess the input image (convert to grayscale and extract features, e.g., HOG)
# gray_image = color.rgb2gray(input_image)
# resized_img = transform.resize(gray_image, desired_size)
# feature_vector = feature.hog(resized_img, pixels_per_cell=(8, 8))
#
# # Reshape the feature vector to match the training data (if necessary)
# feature_vector = feature_vector.reshape(1, -1)
#
# # Make a prediction using the KNN model
# predicted_label = knn.predict(feature_vector)
#
# # Display the predicted label
# print("Predicted Label:", predicted_label[0])


# In[11]:


true_labels = ["actinic keratosis","basal cell carcinoma","basal cell carcinoma","dermatofibroma","melanoma","nevus","pigmented benign keratosis","seborrheic keratosis"]
predicted_labels = ["nevus","basal cell carcinoma","basal cell carcinoma","dermatofibroma","melanoma","nevus","pigmented benign keratosis","basal cell carcinoma"]


# In[12]:


accuracy_val = accuracy_score(true_labels, predicted_labels)
print("Validation Accuracy Percentage:", accuracy_val * 100, "%")


# In[13]:


import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


# In[15]:


true_labels = ["actinic keratosis", "basal cell carcinoma", "basal cell carcinoma", "dermatofibroma", "melanoma", "nevus", "pigmented benign keratosis", "seborrheic keratosis"]
predicted_labels = ["nevus", "basal cell carcinoma", "basal cell carcinoma", "dermatofibroma", "melanoma", "nevus", "pigmented benign keratosis", "basal cell carcinoma"]

# Create a list of unique class labels for both true and predicted labels
classes = list(set(true_labels + predicted_labels))

# Calculate the confusion matrix
confusion_mat = confusion_matrix(true_labels, predicted_labels, labels=classes)

# Create a visualization of the confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(confusion_mat, interpolation='nearest', cmap=plt.get_cmap('Blues'))
plt.title('Confusion Matrix')
plt.colorbar()

# Set tick labels and axis labels
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

# Display the values in the cells
for i in range(len(classes)):
    for j in range(len(classes)):
        plt.text(j, i, str(confusion_mat[i, j]), horizontalalignment='center', color='white')

plt.ylabel('True Labels')
plt.xlabel('Predicted Labels')
plt.show()








# In[ ]:




