import csv

file1 = open('cnn_features.txt','r')
file2 = open('cnn_features.csv','w')
f_data = file1.readlines()
newFileWriter = csv.writer(file2)
newFileWriter.writerow(['labels','features'])
for line in f_data:
    data = line.strip().split("\t")
    if(len(data)==2):
        newFileWriter.writerow([data[1],data[0]])
        
