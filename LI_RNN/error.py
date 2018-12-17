import gensim, logging, os
from gensim.models import Word2Vec
import nltk
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

def main():
    file1 = open("Codemixed.txt","r+")
    file_data = file1.readlines()
    tag = []
    word = []
    sent = []
    tags= []
    count = 0
    count1 = 0
    for line in file_data:
        count+=1
        data = line.split("\t")
        if(len(data)!=3):
            #print(len(data),count)
            count1+=1
            sent.append(word)
            tags.append(tag)
            tag= []
            word = [] 
        else:    
            word.append(data[0])
            tag.append(data[1])
            
    print(sent)        
    word_fname = 'word.model'
    tag_fname = 'tag.model'
    model = Word2Vec(sent,100, min_count=1)
    model.save(word_fname)
    #model2 = Word2Vec(tags, 200, min_count=1)
    #model2.save(tag_fname)
    #print(len(sent), len(tags), tags[0][0], model[tags[0][0]])
        
if __name__=="__main__":
       main()
