from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import numpy as np,pandas as pd
import os

data = pd.read_csv(os.path.join("templates", "Training.csv"))
df = pd.DataFrame(data)
cols = df.columns
cols = cols[:-1]
x = df[cols]
y = df['prognosis']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

print ("DecisionTree")
dt = DecisionTreeClassifier()
clf_dt=dt.fit(x_train,y_train)

# with open('templates/Testing.csv', newline='') as f:
#         reader = csv.reader(f)
#         symptoms = next(reader)
#         symptoms = symptoms[:len(symptoms)-1]

indices = [i for i in range(132)]
symptoms = df.columns.values[:-1]

dictionary = dict(zip(symptoms,indices))

def dosomething(symptom):
    user_input_symptoms = symptom
    user_input_label = [0 for i in range(132)]
    for i in user_input_symptoms:
        idx = dictionary.get(i)  # Use dictionary.get(i) instead of dictionary[i] to handle cases where the symptom is not in the dictionary
        if idx is not None:  # Check if the symptom is valid
            user_input_label[idx] = 1

    user_input_label = np.array(user_input_label)
    user_input_label = user_input_label.reshape(1, -1)  # Reshape the input for prediction
    probas = dt.predict_proba(user_input_label)  # Get probability estimates for each class
    top_classes = np.argsort(-probas, axis=1)[:, :3]  # Get indices of top 3 classes based on probabilities
    top_diseases = [clf_dt.classes_[idx] for idx in top_classes[0]]  # Get the disease names corresponding to the top classes
    return top_diseases


# print(dosomething(['headache','muscle_weakness','puffy_face_and_eyes','mild_fever','skin_rash']))
# prediction = []
# for i in range(7):
#     pred = dosomething(['headache'])   
#     prediction.append(pred) 
# print(prediction)


# --- NAIVE BAYES ---
# from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import GaussianNB
# import numpy as np
# import pandas as pd
# import os

# # Load the dataset
# data = pd.read_csv(os.path.join("templates", "Training.csv"))
# df = pd.DataFrame(data)

# # Prepare the data
# x = df.drop('prognosis', axis=1)
# y = df['prognosis']
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# # Train the Naive Bayes model
# print("Naive Bayes")
# nb = GaussianNB()
# clf_nb = nb.fit(x_train, y_train)

# # Define a dictionary mapping symptoms to their indices
# indices = [i for i in range(132)]
# symptoms = df.columns.values[:-1]
# dictionary = dict(zip(symptoms, indices))

# # Define a function to predict diseases based on symptoms using Naive Bayes
# def predict_diseases(symptom):
#     user_input_symptoms = symptom
#     user_input_label = [0 for i in range(132)]
#     for i in user_input_symptoms:
#         idx = dictionary.get(i)
#         if idx is not None:
#             user_input_label[idx] = 1

#     user_input_label = np.array(user_input_label).reshape(1, -1)
#     probas = nb.predict_proba(user_input_label)
#     top_classes = np.argsort(-probas, axis=1)[:, :3]
#     top_diseases = [clf_nb.classes_[idx] for idx in top_classes[0]]
#     return top_diseases

# # Test the function
# print(predict_diseases(['headache', 'muscle_weakness', 'puffy_face_and_eyes', 'mild_fever', 'skin_rash']))

# --- RANDOM FOREST ---
# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier
# import numpy as np
# import pandas as pd
# import os

# # Load the dataset
# data = pd.read_csv(os.path.join("templates", "Training.csv"))
# df = pd.DataFrame(data)

# # Prepare the data
# x = df.drop('prognosis', axis=1)
# y = df['prognosis']
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# # Train the KNN model
# print("k-Nearest Neighbors")
# knn = KNeighborsClassifier()
# clf_knn = knn.fit(x_train, y_train)

# # Define a dictionary mapping symptoms to their indices
# indices = [i for i in range(132)]
# symptoms = df.columns.values[:-1]
# dictionary = dict(zip(symptoms, indices))

# # Define a function to predict diseases based on symptoms using KNN
# def predict_diseases_knn(symptom):
#     user_input_symptoms = symptom
#     user_input_label = [0 for i in range(132)]
#     for i in user_input_symptoms:
#         idx = dictionary.get(i)
#         if idx is not None:
#             user_input_label[idx] = 1

#     user_input_label = np.array(user_input_label).reshape(1, -1)
#     top_classes = clf_knn.predict_proba(user_input_label)
#     top_indices = np.argsort(-top_classes, axis=1)[:, :3]
#     top_diseases = [clf_knn.classes_[idx] for idx in top_indices[0]]
#     return top_diseases

# # Test the function
# print(predict_diseases_knn(['headache', 'muscle_weakness', 'puffy_face_and_eyes', 'mild_fever', 'skin_rash']))

# --- KNN ---
# from sklearn.model_selection import train_test_split
# from sklearn.neighbors import KNeighborsClassifier
# import numpy as np
# import pandas as pd
# import os

# # Load the dataset
# data = pd.read_csv(os.path.join("templates", "Training.csv"))
# df = pd.DataFrame(data)

# # Prepare the data
# x = df.drop('prognosis', axis=1)
# y = df['prognosis']
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

# # Train the KNN model
# print("k-Nearest Neighbors")
# knn = KNeighborsClassifier()
# clf_knn = knn.fit(x_train, y_train)

# # Define a dictionary mapping symptoms to their indices
# indices = [i for i in range(132)]
# symptoms = df.columns.values[:-1]
# dictionary = dict(zip(symptoms, indices))

# # Define a function to predict diseases based on symptoms using KNN
# def predict_diseases_knn(symptom):
#     user_input_symptoms = symptom
#     user_input_label = [0 for i in range(132)]
#     for i in user_input_symptoms:
#         idx = dictionary.get(i)
#         if idx is not None:
#             user_input_label[idx] = 1

#     user_input_label = np.array(user_input_label).reshape(1, -1)
#     top_classes = clf_knn.predict_proba(user_input_label)
#     top_indices = np.argsort(-top_classes, axis=1)[:, :3]
#     top_diseases = [clf_knn.classes_[idx] for idx in top_indices[0]]
#     return top_diseases

# # Test the function
# print(predict_diseases_knn(['headache', 'muscle_weakness', 'puffy_face_and_eyes', 'mild_fever', 'skin_rash']))
