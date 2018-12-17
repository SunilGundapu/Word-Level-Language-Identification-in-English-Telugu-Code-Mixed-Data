import gensim, logging, os
from gensim.models import Word2Vec
import nltk
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
def word2vec(input_file):
    st = []
    with open(input_file) as f:
        for line in f:
            list1=nltk.word_tokenize(line.strip())
            st.append(list1)
                 
    fname = 'input.model'
    model = Word2Vec(st,200, min_count=1)
    model.save(fname)
    return model
   
word2vec('input.txt')
            