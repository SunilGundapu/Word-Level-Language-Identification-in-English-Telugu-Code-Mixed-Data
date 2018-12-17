import re

def wordpref(word):
    k1,k2,k3 = 'null','null', 'null' 
    if(len(word)==1):
        k1 = word
        k2 = 'null'
        k3 = 'null'
    elif(len(word)==2):
        k1 = word[0:1]
        k2 = word[0:2]
        k3 = 'null'
    elif(len(word)>=3):
        k1 = word[0:1]
        k2 = word[0:2]
        k3 = word[0:3]
    return k1,k2,k3
    
def wordsuf(word):
    k1,k2,k3 = 'null','null', 'null' 
    if(len(word)==1):
        k1 = 'null'
        k2 = 'null'
        k3 = word
    elif(len(word)==2):
        k1 = 'null'
        k2 = word[-2:]
        k3 = word[-3:]
    elif(len(word)>=3):
        k1 = word[-1:]
        k2 = word[-2:]
        k3 = word[-3:]
    return k1,k2,k3
    
def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

voc = []
file1 = open("Codemixed.txt","r")
f_data = file1.readlines()
file1 = open("te.txt","w")
file2 = open("en.txt","w")
file3 = open("ne.txt","w")
file4 = open("univ.txt","w")
for line in f_data:
    words = line.strip().split("\t")
    if(len(words)==3):
        voc.append(words[0])
        pref = wordpref(words[0])
        suf = wordsuf(words[0])
        start_cap = words[0].isupper()
        end_cap = words[-1].isupper()
        start_dig = words[0].isdigit()
        end_dig = words[-1].isdigit()
        containdig = hasNumbers(words[0])
        speacial = "Invalid" if re.match("^[a-zA-Z0-9_]*$", words[0]) else "Valid"
        if(words[1]=='te'):
            file1.write(words[0]+" "+words[2]+" "+pref[0]+" "+pref[1]+" "+pref[2]+" "+suf[0]+" "+suf[1]+" "+suf[2]+" "+str(start_cap)+" "+str(end_cap)+" "+str(start_dig)+" "+str(end_dig)+" "+str(containdig)+" "+str(speacial)+"\n")
        elif(words[1]=='en'):
            file2.write(words[0]+" "+words[2]+" "+pref[0]+" "+pref[1]+" "+pref[2]+" "+suf[0]+" "+suf[1]+" "+suf[2]+" "+str(start_cap)+" "+str(end_cap)+" "+str(start_dig)+" "+str(end_dig)+" "+str(containdig)+" "+str(speacial)+"\n")
        elif(words[1]=='ne'):
            file3.write(words[0]+" "+words[2]+" "+pref[0]+" "+pref[1]+" "+pref[2]+" "+suf[0]+" "+suf[1]+" "+suf[2]+" "+str(start_cap)+" "+str(end_cap)+" "+str(start_dig)+" "+str(end_dig)+" "+str(containdig)+" "+str(speacial)+"\n")
        elif(words[1]=='univ'):
            file4.write(words[0]+" "+words[2]+" "+pref[0]+" "+pref[1]+" "+pref[2]+" "+suf[0]+" "+suf[1]+" "+suf[2]+" "+str(start_cap)+" "+str(end_cap)+" "+str(start_dig)+" "+str(end_dig)+" "+str(containdig)+" "+str(speacial)+"\n")
        
print(len(set(voc)))


