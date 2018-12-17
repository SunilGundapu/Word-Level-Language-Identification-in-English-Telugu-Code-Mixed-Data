from textblob import TextBlob
from textblob.classifiers import NaiveBayesClassifier
from textblob.classifiers import MaxEntClassifier
from textblob.classifiers import DecisionTreeClassifier
import nltk
#import sys
#reload(sys)
#sys.setdefaultencoding('utf-8')

words = []
tags = []
train = []
test = []
test_tags = []
pred_tags = []
temp = ()
count=0

f = open("Codemixed.txt","r")
file_data = f.readlines()
for line in file_data:
    a = line.split("\t")
    if(len(a)==3):
        words.append(a[0])
        tags.append(a[1])

print(len(words), len(tags))

for i in range(1000):
    if(i<800):
        temp=(words[i],tags[i])
        train.append(temp)
    else:
        temp=(words[i],tags[i])
        test.append(temp)
print(train)
print(test)

naive = NaiveBayesClassifier(train)
dtc = DecisionTreeClassifier(train)
mec = MaxEntClassifier(train)

print("NaiveBayesClassifier Accuracy: {0}".format(naive.accuracy(test)))
print("DecisionTreeClassifier Accuracy: {0}".format(dtc.accuracy(test)))
print("MaxEntClassifier Accuracy: {0}".format(mec.accuracy(test)))



cl = NaiveBayesClassifier(train)
print("NaiveBayesClassifier Accuracy: {0}".format(cl.accuracy(test)))
for i in range(0,len(test)):
    tag = cl.classify(test[i])
    pred_tags.append(tag)
    if(tag == test_tags[i]):
        count+=1
print(len(pred_tags),len(test_tags))
print(count)
