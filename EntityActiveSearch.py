
# coding: utf-8

# In[163]:

import activeSearchInterface as asI
import scipy.sparse as ss
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel,linear_kernel
#import re
#from simhash import Simhash, SimhashIndex#pip install simhash
from hashes.simhash import simhash#faster than above
import string
from random import randint
import json
#from nltk.util import ngrams
from MyNgrams import ngrams
import cPickle as pickle

from kernelAS import kernel_AS


class EntityAS():
    
    def __init__ (self,threshold=0.98):
        self.duplicate_threshold=threshold#needed for simhash but not for not minhash
        self.activeSearch = asI.shariAS() ##slow
        #self.activeSearch = asI.kernelAS() ##fast
        
        self.prev_corpus=[]#previously labeled data. List of touples (id,text) where id is external id
        self.prev_labels=[]#labels for previously labeled data.
        self.vectorizer=None
        self.curr_corpus=[]
        self.id_to_idx={}#maps external id (e.g. AdId) to internal corpus index
        self.index_map={}#maps active search matrix indices to indices of corpus
        self.index_map_reverse={}#maps indices from corpus to active search matrix indices
        self.near_duplicates={}#for indices in corpus points to indices of near duplicates        
        self.num_messages=-1
        self.start_idx=0
        #variables needed to clean text
        self.chars=string.punctuation+' '
        #self.remove = dict((ord(char), u' ') for char in chars)#this is necessary for unicode strings 

        #TODO
    def save(self,filename):
        #save labels and objects so system can restart with them
        pickle.dump(self.prev_corpus,filename+'_labels.pkl')
        
        #TODO
    def load(self,filename):
        prev_corpus = pickle.load(filename+'_labels.pkl')
        
        #TODO
    def restartActiveSearch(self):#back up information, restart with new similarity function
        #save labels
        labels_backup=self.activeSearch.labels

    def hashing(self,text):
        n_grams = [x for x in ngrams(text.split(),5)]
        if len (n_grams) > 0:
            return min([int(hashlib.sha256(gram).hexdigest(),16) for gram in n_grams])
        else:
            int(hashlib.sha256("").hexdigest(),16)
        
    def returnWeights(self):
        weights=[0.0]*self.num_messages
        for key in self.index_map:
            for idx in self.near_duplicates[self.index_map[key]]:
                weights[idx]=self.activeSearch.f[key]
        #return list of (ids,weights)
        return [(self.corpus[i][0],w) for i,w in enumerate(weights)]
        
        
    def initializeAS(self,X):
        #self.activeSearch = asI.kernelAS() ##fast
        self.activeSearch = asI.shariAS() ##slow
        self.activeSearch.initialize(np.array(X.todense()))
        
        
    def setCountVectorizer(self,binary=True,ngram_range=(1,1),max_df=0.9,min_df=0.005):
        self.vectorizer=CountVectorizer(analyzer='word',binary=binary,ngram_range=ngram_range,max_df=max_df,min_df=min_df)
    
    def setTfidf(self):
        self.vectorizer=TfidfVectorizer(norm='l2',stop_words='english',analyzer='word',max_df=0.95,min_df=0.005)
        

    def get_random_message(self):
        mid=randint(0,self.num_messages)
        return self.curr_corpus[mid][0]
        
    def get_text(self,idx):
        return self.corpus[idx][1]
    
    def get_id(self,idx):
        return self.corpus[idx][0]
    
    def get_next_ids(self):
        m=self.activeSearch.getNextMessage()
        return [myAS.corpus[x][0] for x in self.near_duplicates[self.index_map[m]]]
    
    def get_next_texts(self):
        m=self.activeSearch.getNextMessage()
        return [myAS.corpus[x][1] for x in self.near_duplicates[self.index_map[m]]]
    
    def get_next(self):
        m=self.activeSearch.getNextMessage()
        return [myAS.corpus[x] for x in self.near_duplicates[self.index_map[m]]]
            
    def decide_startingpoint(self,id):
        myAS.activeSearch.firstMessage(self.index_map_reverse[self.id_to_idx[id]])        
        
    def startJsonStr(self,jsonstring,textfield="text"):
        obj=json.loads(jsonstring)
        self.startJsonObj(obj,textfield)
        
    def startJsonObj(self,obj,textfield="text"):
        corpus=[(key,obj[key][textfield]) for key in obj]
        
    def nextActiveSearch(self):
        #preserve labels
        labels_backup=self.activeSearch.labels

    def next_AS(self,corpus):
        labels=[x for x in self.activeSearch.labels[self.start_idx:] if x !=-1]
        new_labeled_indices=[i+self.start_idx for i,x in enumerate(self.activeSearch.labels[self.start_idx:]) if x !=-1]        
        save_corpus=[self.curr_corpus[i+self.start_idx] for i,x in enumerate(self.activeSearch.labels[self.start_idx:]) if x !=-1]
        
    def newActiveSearch(self,corpus,labeled_corpus=[],labels=[],starting_points=None):
        """
        corpus --> list of touples (id,text) where id is external id
        """
        num_labels = len(labels)
        if num_labels != len(labeled_corpus):
            raise Exception ("Number of lables and number of previously labeled objects does not match")
        if num_labels > 0:
            self.prev_corpus.extend(labeled_corpus)
            self.prev_labels.extend(labels)
        
        #initialise withprevious information
        start_idx=len(self.prev_labels)    
        self.start_idx=start_idx  

        #get map from external id to internal index
        self.id_to_idx={}#maps external id (e.g. AdId) to internal index
        for i,el in enumerate(corpus):
            self.id_to_idx[el[0]]=i+start_idx #do not include indices pointing to already labeled objects from previous AS

        self.curr_corpus=corpus
        self.num_messages=len(corpus)
        #hashed=[simhash(tup[1].lower()) for i,tup in enumerate(corpus)]
        hashed=[self.hashing(tup[1].lower()) for i,tup in enumerate(corpus)]#minhash
        alt_corpus=[]
        self.index_map={}#maps active search matrix indices to indices of curr_corpus
        self.index_map_reverse={}#maps indices from corpus to active search matrix indices
        self.near_duplicates={}#for indices in corpus points to indices of near duplicates 
        
        flag=True
        
        # reduce size of corpus passed on to AS by removing near-duplicates
        # create a mapping from original corpus passed to function to the new, smaller alt_copus 
        # where near-duplicates have been removed
        count=start_idx#initialize with number of previously labeled objects
        for i,x in enumerate(hashed):
            for y in alt_corpus:
                #if x.similarity(y[1]) > self.duplicate_threshold:#simhash
                if x==y[1]:#minhash
                    self.near_duplicates[y[0]].append(i)
                    self.index_map_reverse[i]=self.index_map_reverse[y[0]]
                    flag=False
                    break
            if flag:
                alt_corpus.append((i,x))
                self.index_map[count]=i
                self.index_map_reverse[i]=count
                count+=1
                self.near_duplicates[i]=[i]
            flag=True

        #featurize
        #self.setTfidf()
        self.setCountVectorizer()
        self.X=self.vectorizer.fit_transform(self.prev_corpus + [self.curr_corpus[y[0]][1] for y in alt_corpus])
        #Xaff=rbf_kernel(X,gamma=1)


        self.labels=labels+[-1]*len(alt_corpus)
        #kernel_AS(X, labels=self.labels, num_initial=1, num_eval=1, pi=0.05, eta=0.5, w0=None, init_pt=None, 
        #          verbose=True, all_fs=False, sparse=False, tinv=False)
        if starting_points is not None:
            if not isinstance(starting_points, list):
                starting_points = [starting_points]
            for x in starting_points:
                self.labels[self.index_map_reverse[self.id_to_idx[x]]]=1
        elif num_labels==0:
            raise Exception ("No start point and no labels provided")
        #print self.X.shape
        #print len(self.labels)
        #kernel_AS(self.X, labels=self.labels, num_initial=1, num_eval=1, pi=0.05, eta=0.5, w0=None, init_pt=None, 
        #          verbose=True, all_fs=False, sparse=False, tinv=False)
        #TODO
        #initialize and set labels
        
        #sheriAS needs an affinity matrix :( 
        Xaff=linear_kernel(self.X)
        self.activeSearch.initialize(Xaff)
        myAS.decide_startingpoint(starting_points[0])
        
    def start_DEPRECATED(self,corpus):#assumes cleaned text for now
        """
        corpus --> list of touples (id,text)
        """
    
        self.id_to_idx={}
        for i,el in enumerate(corpus):
            self.id_to_idx[el[0]]=i
    
        self.corpus=corpus
        self.num_messages=len(corpus)
        hashed=[simhash(tup[1].lower()) for i,tup in enumerate(corpus)]
        alt_corpus=[]
        self.index_map={}
        self.near_duplicates={}
        self.index_map_reverse={}
        flag=True
        count=0

        #calculate simhashes
        for i,x in enumerate(hashed):
            for y in alt_corpus:
                if x.similarity(y[1]) > self.duplicate_threshold:
                    self.near_duplicates[y[0]].append(i)
                    self.index_map_reverse[i]=self.index_map_reverse[y[0]]
                    flag=False
                    break
            if flag:
                alt_corpus.append((i,x))
                self.index_map[count]=i
                self.index_map_reverse[i]=count
                count+=1
                self.near_duplicates[i]=[i]
                
            flag=True
            
        #featurize
        #self.setTfidf()
        self.setCountVectorizer()
        X=self.vectorizer.fit_transform([self.corpus[y[0]][1] for y in alt_corpus])
        #Xaff=rbf_kernel(X,gamma=1)
        Xaff=linear_kernel(X)
        #sheriAS needs an affinity matrix :( 
        self.initializeAS(Xaff)
        self.activeSearch.initialize(Xaff)


# In[116]:


corpus1=[] 
corpus2=[] 
chuncksize=1000
for i,line in enumerate(open('US_Canada_dedup_clean_text_training.txt','r')):
    if i < chuncksize:
        corpus1.append(line.rstrip().lower())
    else:
        
        if i < chuncksize*2:
            corpus2.append(line.rstrip().lower())
        else:
            break


# In[3]:

#%load_ext cythonmagic


# In[58]:

len(corpus2)


# In[164]:

myAS=EntityAS()
myAS.newActiveSearch([(i,x) for i,x in enumerate(corpus2)],starting_points=[534])


# In[123]:




# In[148]:

mid=myAS.get_random_message()
corpus2[mid]


# In[149]:

mid


# In[66]:




# In[87]:

#myAS.decide_startingpoint(215)
myAS.decide_startingpoint(mid)


# In[13]:




# In[112]:

ids=myAS.get_next_ids()
for x in ids:
    print corpus2[x]


# In[111]:

myAS.activeSearch.boringMessage()


# In[109]:

myAS.activeSearch.boringMessage()


# In[112]:




# In[99]:

import re
from simhash import Simhash, SimhashIndex
import string
from nltk.util import ngrams
from MyNgrams import ngrams as myngrams
from hashes.simhash import simhash#faster than above
import hashlib

chars=string.punctuation+' '
duplicate_threshold=0.98



def get_features(s,n= 3):
    #s = re.sub(r'[^\w]+', '', s)#swaped out with string translate function to make this faster
    s=s.translate(None,chars)
    return [s[i:i + n] for i in range(max(len(s) - n + 1, 1))]

def try_indexing_1():
    
    #remove = dict((ord(char), u' ') for char in chars)#this is necessary for unicode strings 
    #sentence="How are you? I Am fine. blar blar blar blar blar Thanks.3"
    #get_features(sentence)
    hashed=[ Simhash(get_features(v.lower())) for i, v in enumerate(corpus1)]
    #index = SimhashIndex(objs, k=3)
    alt_corpus=[]
    index_map={}
    near_duplicates={}
    index_map_reverse={}
    flag=True
    count=0

    #calculate simhashes
    for i,x in enumerate(hashed):
        for y in alt_corpus:
            if 1.0-float(x.distance(y[1]))/64.0 > duplicate_threshold:
                near_duplicates[y[0]].append(i)
                index_map_reverse[i]=index_map_reverse[y[0]]
                flag=False
                break
        if flag:
            alt_corpus.append((i,x))
            index_map[count]=i
            index_map_reverse[i]=count
            count+=1
            near_duplicates[i]=[i]

        flag=True
    print len(alt_corpus)
    
def try_indexing_2():
    #remove = dict((ord(char), u' ') for char in chars)#this is necessary for unicode strings 
    #sentence="How are you? I Am fine. blar blar blar blar blar Thanks.3"
    #get_features(sentence)
    hashed=[simhash(v.lower()) for i, v in enumerate(corpus1)]
    #index = SimhashIndex(objs, k=3)
        
    #remove = dict((ord(char), u' ') for char in chars)#this is necessary for unicode strings 
    #sentence="How are you? I Am fine. blar blar blar blar blar Thanks.3"
    #get_features(sentence)
    #index = SimhashIndex(objs, k=3)
    alt_corpus=[]
    index_map={}
    near_duplicates={}
    index_map_reverse={}
    flag=True
    count=0

    #calculate simhashes
    for i,x in enumerate(hashed):
        for y in alt_corpus:
            if x.similarity(y[1]) > duplicate_threshold:
                near_duplicates[y[0]].append(i)
                index_map_reverse[i]=index_map_reverse[y[0]]
                flag=False
                break
        if flag:
            alt_corpus.append((i,x))
            index_map[count]=i
            index_map_reverse[i]=count
            count+=1
            near_duplicates[i]=[i]

        flag=True
    print len(alt_corpus)

def try_indexing_3():
        #remove = dict((ord(char), u' ') for char in chars)#this is necessary for unicode strings 
    #sentence="How are you? I Am fine. blar blar blar blar blar Thanks.3"
    #get_features(sentence)
    hashed=[myhashing((v.lower())) for v in corpus1]
    #index = SimhashIndex(objs, k=3)
        
    #remove = dict((ord(char), u' ') for char in chars)#this is necessary for unicode strings 
    #sentence="How are you? I Am fine. blar blar blar blar blar Thanks.3"
    #get_features(sentence)
    #index = SimhashIndex(objs, k=3)
    alt_corpus=[]
    index_map={}
    near_duplicates={}
    index_map_reverse={}
    flag=True
    count=0

    #calculate simhashes
    for i,x in enumerate(hashed):
        for y in alt_corpus:
            if x == y[1]:
                near_duplicates[y[0]].append(i)
                index_map_reverse[i]=index_map_reverse[y[0]]
                flag=False
                break
        if flag:
            alt_corpus.append((i,x))
            index_map[count]=i
            index_map_reverse[i]=count
            count+=1
            near_duplicates[i]=[i]

        flag=True
    print len(alt_corpus)
    
    #most_common_words = most_common_words_list.most_common_words
    #4, 6 gives good cross-phone number matching
    #objs=[ nltkhashing(v.lower()) for i, v in enumerate(corpus1)]
    #return objs
  


def myhashing(text):
    n_grams = [x for x in myngrams(text.split(),5)]
    return min([int(hashlib.sha256(gram).hexdigest(),16) for gram in n_grams])
    

    #return str(min([int(hashlib.sha256(' '.join(gram)).hexdigest(),16) for gram in n_grams]))
    
def try_deduplication():
    chars=string.punctuation+' '
    objs=[(str(i), Simhash(get_features(v.lower()))) for i, v in enumerate(corpus1)]


# In[86]:

text = "how do these look and what does the whitespace do?"
[x for x in ngrams(text.split(),5)]


# In[100]:

import cProfile
#cProfile.run('try_indexing_1()')#9.9 -10 seconds
#cProfile.run('try_indexing_2()')#6.6-7seconds
cProfile.run('try_indexing_3()')#6.6-7seconds
#cProfile.run('try_indexing_4()')#6.6-7seconds
#objs=try_indexing_3()
#objs=try_indexing_4()


# In[55]:

from MyNgrams import ngrams as myngrams
text="hello world please try this one"
n_grams = list(myngrams(text.split(),4))
n_grams


# In[151]:

objs[999]


# In[120]:

import cProfile
cProfile.run('try_indexing_2()')
cProfile.run('try_indexing_3()')

