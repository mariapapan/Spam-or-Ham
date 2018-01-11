import os
import numpy as np
from collections import Counter
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score

def make_dictionary(train_dir):
    emails = [os.path.join(train_dir,f) for f in os.listdir(train_dir)]
    all_words = []
    for mail in emails:
        with open(mail) as m: #the "with" statement is used when after the execution of this code we want to close the file
            for i,line in enumerate(m):
                if (i == 2): #Body of the email is in 3rd line of the text file
                    words = line.split()
                    all_words += words

    dictionary = Counter(all_words) #Counter creates a dictionary with the values and how many times equivalent values are added
    list_to_remove = list(dictionary.keys())
    #to keep only words for the dictionary but not those with one letter like "I"
    for item in list_to_remove:
        if (item.isalpha() is False) or len(item) == 1:
           del dictionary[item]
    dictionary = dictionary.most_common(3000)
    
    return dictionary
    
def extract_features(mail_dir): 
    files = [os.path.join(mail_dir,fi) for fi in os.listdir(mail_dir)]
    features_matrix = np.zeros((len(files),3000))
    docID = 0;
    for fil in files:
      with open(fil) as fi:
        for i,line in enumerate(fi):
          if i == 2:
            words = line.split()
            for word in words:
              wordID = 0
              for i,d in enumerate(dictionary):
                if d[0] == word:
                  wordID = i
                  features_matrix[docID,wordID] = words.count(word)
        docID = docID + 1     
    
    return features_matrix

#Create a dictionary of words and its frequency

train_dir = 'C:/Users/Coko/PythonProjects/train_mails'
dictionary = make_dictionary(train_dir)

#Prepare feature vectors per training mail and its labels

train_labels = np.zeros(702)
train_labels[351:701] = 1 #second half of mails are spam mails and got label "1"
train_matrix = extract_features(train_dir)

#Training:
# 1. Naive Bayes Classifier
# 2. SVM
# 3. Decision trees 
# 4. k-NN  
# 5. Deep learning

model1 = MultinomialNB()
model2 = LinearSVC()
model3 = DecisionTreeClassifier()
model4 = KNeighborsClassifier(3)
model5 = MLPClassifier(alpha=1e-5, hidden_layer_sizes=(5, 2))
model1.fit(train_matrix, train_labels)
model2.fit(train_matrix, train_labels)
model3.fit(train_matrix, train_labels)
model4.fit(train_matrix, train_labels)
model5.fit(train_matrix, train_labels)

#Test the unseen mails for Spam

test_dir = 'C:/Users/Coko/PythonProjects/test_mails'
test_matrix = extract_features(test_dir)
test_labels = np.zeros(260)
test_labels[130:260] = 1 #spam mails
result1 = model1.predict(test_matrix)
result2 = model2.predict(test_matrix)
result3 = model3.predict(test_matrix)
result4 = model4.predict(test_matrix)
result5 = model5.predict(test_matrix)

print('\n--------Accuracy---------\n')
print('NB model:', accuracy_score(test_labels,result1))
print('SVM model:', accuracy_score(test_labels,result2))
print('Decision Tree model:', accuracy_score(test_labels,result3))
print('KNeighborsClassifier model:', accuracy_score(test_labels,result4))
print('Deep learning model:', accuracy_score(test_labels,result5))

print('\n---------Recall----------\n')
print('NB model:', recall_score(test_labels,result1))
print('SVM model:', recall_score(test_labels,result2))
print('Decision Tree model:', recall_score(test_labels,result3))
print('KNeighborsClassifier model:', recall_score(test_labels,result4))
print('Deep learning model:', recall_score(test_labels,result5))

print('\n--------Precision--------\n')
print('NB model:', precision_score(test_labels,result1))
print('SVM model:', precision_score(test_labels,result2))
print('Decision Tree model:', precision_score(test_labels,result3))
print('KNeighborsClassifier model:', precision_score(test_labels,result4))
print('Deep learning model:', precision_score(test_labels,result5))

