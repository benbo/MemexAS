
# coding: utf-8

import activeSearchInterface as asI
import scipy.sparse as ss
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel,linear_kernel
from sklearn.metrics import roc_auc_score
#import re
#from simhash import Simhash, SimhashIndex#pip install simhash
#from hashes.simhash import simhash#faster than above
from MyNgrams import ngrams
import hashlib
import string
from random import randint
import json
import numpy as np
#from nltk.util import ngrams

import cPickle as pickle


class EntityAS():
    
    def __init__ (self,threshold=0.98):
        self.activeSearch=None
        self.duplicate_threshold=threshold#needed for simhash but not for not minhash        
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
        return [(self.curr_corpus[i][0],w) for i,w in enumerate(weights)]        
        
    def setCountVectorizer(self,binary=True,ngram_range=(1,1),max_df=0.9,min_df=0.005):
        self.vectorizer=CountVectorizer(analyzer='word',binary=binary,ngram_range=ngram_range,max_df=max_df,min_df=min_df)
    
    def setTfidf(self):
        self.vectorizer=TfidfVectorizer(norm='l2',stop_words='english',analyzer='word',max_df=0.95,min_df=0.005)
        

    def get_random_message(self):
        mid=randint(0,self.num_messages)
        return self.curr_corpus[mid][0]
        
    def get_text(self,idx):
        return self.curr_corpus[idx][1]
    
    def get_id(self,idx):
        return self.curr_corpus[idx][0]
    
    def get_next_ids(self):
        m=self.activeSearch.getNextMessage()
        return [self.curr_corpus[x][0] for x in self.near_duplicates[self.index_map[m]]]
    
    def get_next_texts(self):
        m=self.activeSearch.getNextMessage()
        return [self.curr_corpus[x][1] for x in self.near_duplicates[self.index_map[m]]]
    
    def get_next(self):
        m=self.activeSearch.getNextMessage()
        return [self.curr_corpus[x] for x in self.near_duplicates[self.index_map[m]]]
            
    def decide_startingpoint(self,id):
        self.activeSearch.firstMessage(self.index_map_reverse[self.id_to_idx[id]])        
        
    def startJsonStr(self,jsonstring,textfield="text"):
        obj=json.loads(jsonstring)
        self.startJsonObj(obj,textfield)
        
    def startJsonObj(self,obj,textfield="text"):
        corpus=[(key,obj[key][textfield]) for key in obj]

    def next_AS(self,corpus,starting_points=[],dedupe=True,tfidf=False,prevalence=0.1):
        new_labeled_indices=[i+self.start_idx for i,x in enumerate(self.activeSearch.labels[self.start_idx:]) if x !=-1]
        prev_labels=[self.activeSearch.labels[x] for x in new_labeled_indices]
        prev_corpus=[self.curr_corpus[self.index_map[x]] for x in new_labeled_indices]
        self.newActiveSearch(corpus,labeled_corpus=prev_corpus,labels=prev_labels,starting_points=starting_points,dedupe=dedupe,tfidf=tfidf)
        
    def evaluate_scores(self,labels):
        f=np.array(self.activeSearch.f[self.start_idx:])
        return roc_auc_score(np.array(labels), f)
        
    def newActiveSearch(self,corpus,labeled_corpus=[],labels=[],starting_points=[],dedupe=True,tfidf=False,prevalence=0.1):
        """
        corpus --> list of touples (id,text) where id is external id
        """
        num_labels = len(labels)
        if num_labels != len(labeled_corpus):
            raise Exception ("Number of lables and number of previously labeled objects does not match")
        if num_labels > 0:
            self.prev_corpus.extend(labeled_corpus)
            self.prev_labels.extend(labels)
        
        if tfidf:
            self.setTfidf()
        else:
            self.setCountVectorizer()
        
        #initialise with previous information
        start_idx=len(self.prev_labels)    
        self.start_idx=start_idx  

        #get map from external id to internal index for the new corpus 
        self.id_to_idx={}#maps external id (e.g. AdId) to internal index
        for i,el in enumerate(corpus):
            self.id_to_idx[el[0]]=i+start_idx #do not include indices pointing to already labeled objects from previous AS

        self.curr_corpus=corpus
        self.num_messages=len(corpus)

        alt_corpus=[]
        self.index_map={}#maps active search matrix indices to indices of curr_corpus
        self.index_map_reverse={}#maps indices from corpus to active search matrix indices
        self.near_duplicates={}#for indices in corpus points to indices of near duplicates 
        
        if dedupe:
            # reduce size of corpus passed on to AS by removing near-duplicates
            # create a mapping from original corpus passed to function to the new, smaller alt_copus 
            # where near-duplicates have been removed

            flag=True
            #hashed=[simhash(tup[1].lower()) for i,tup in enumerate(corpus)]#simhash
            hashed=[self.hashing(tup[1].lower()) for i,tup in enumerate(corpus)]#minhash
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
                    self.near_duplicates[i]=[i]
                    count+=1
                flag=True
            alt_len=len(alt_corpus)
            #featurize
            self.X=self.vectorizer.fit_transform([x[1] for x in self.prev_corpus] + [corpus[y[0]][1] for y in alt_corpus])
        else:
            count=start_idx
            for i in range(self.num_messages):
                self.index_map[count]=i
                self.index_map_reverse[i]=count
                self.near_duplicates[i]=[i]
                count+=1
            alt_len=self.num_messages
            #featurize
            self.X=self.vectorizer.fit_transform([x[1] for x in self.prev_corpus] + [y[1] for y in corpus])


        #self.labels=self.prev_labels+[-1]*alt_len
        #kernel_AS(X, labels=self.labels, num_initial=1, num_eval=1, pi=0.05, eta=0.5, w0=None, init_pt=None, 
        #          verbose=True, all_fs=False, sparse=False, tinv=False)
   
        
        #print self.X.shape
        #print len(self.labels)
        #kernel_AS(self.X, labels=self.labels, num_initial=1, num_eval=1, pi=0.05, eta=0.5, w0=None, init_pt=None, 
        #          verbose=True, all_fs=False, sparse=False, tinv=False)
        #TODO
        #initialize and set labels
        
        #sheriAS needs an affinity matrix :( 
        #Xaff=rbf_kernel(X,gamma=1)
        #Xaff=linear_kernel(self.X)
        #self.activeSearch = asI.shariAS() ##slow
        params=asI.Parameters(pi=prevalence,verbose=False)
        self.activeSearch = asI.kernelAS(params=params) ##fast
        self.activeSearch.initialize(self.X.transpose())
        if len(starting_points)==0:
            if len(self.prev_labels)==0:
                raise Exception ("No start point and no labels provided")
        else:
            self.decide_startingpoint(starting_points[0])
            for x in starting_points[1:]:
                self.activeSearch.setLabel(self.index_map_reverse[self.id_to_idx[i]],1)
        for i,x in enumerate(self.prev_labels):
            self.activeSearch.setLabel(i,x)
from os import listdir
from os.path import isfile, join
import random
#myAS=EntityAS()

def test():
    mypath='test'
    files = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
    overall_results=[]
    for f in files:
        results=[]
        text=[(i,line.rstrip()) for i,line in enumerate(open(join(mypath,f),'r'))]
        labels=[1]*100+[0]*900
        startp=random.randint(0,99)
        myAS=EntityAS()
        myAS.newActiveSearch(text,starting_points=[startp],tfidf=True,dedupe=False)
        for x in range(len(text)-2):
            results.append(myAS.evaluate_scores(labels))
            l=myAS.get_next_ids()
            idx=l[0]
            if labels[idx]==1:
                myAS.activeSearch.interestingMessage()
            else:
                myAS.activeSearch.boringMessage()
        overall_results.append(results)
    return overall_results

def test_transfer():
    mypath='test'
    files = [ f for f in listdir(mypath) if isfile(join(mypath,f)) ]
    overall_results=[]
    myAS=EntityAS()
    flag=True
    for f in files:
        results=[]
        text=[(i,line.rstrip()) for i,line in enumerate(open(join(mypath,f),'r'))]
        labels=[1]*100+[0]*900
        startp=random.randint(0,99)
        if flag:
            myAS.newActiveSearch(text,starting_points=[startp],tfidf=True,dedupe=False)
        else:
            myAS.next_AS(text,tfidf=True,dedupe=False)
        for x in range(len(text)-2):
            results.append(myAS.evaluate_scores(labels))
            l=myAS.get_next_ids()
            idx=l[0]
            if labels[idx]==1:
                myAS.activeSearch.interestingMessage()
            else:
                myAS.activeSearch.boringMessage()
        overall_results.append(results)
    return overall_results
#notransfer=test()
#f = open('imdb_notransfer.txt','wb')
#pickle.dump(notransfer,f)
#f.close()
transfer=test_transfer()
f = open('imdb_transfer.txt','wb')
pickle.dump(transfer,f)
f.close()

        
