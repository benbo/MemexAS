# coding: utf-8
import activeSearchInterface as asI
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing

from scipy.spatial.distance import pdist, squareform
from scipy import sparse

from MyNgrams import ngrams
import hashlib
from random import randint



class MemexAS():
    
    def __init__ (self,threshold=0.98):
        self.activeSearch=None
        self.duplicate_threshold=threshold#needed for simhash but not for not minhash        
        self.prev_corpus=[]#previously labeled data. List of touples (id,text) where id is external id
        self.prev_labels=[]#labels for previously labeled data.
        self.vectorizer=None
        self.curr_corpus=[]
        self.id_to_idx={}#maps external id (e.g. AdId) to internal corpus index
        self.hashlookup={}
        self.num_messages=-1
        self.start_idx=0
        self.extendedVocabulary=set()
        self.Xsparse=None
        self.scalefactor=0
        self.dedupe=True
        self.tfidf=False
        self.prevalence=0.1
        self.eta=0.5
        self.dimred=False
        self.sparse=True


    def interestingMessage(self):
        if len(self.activeSearch.unlabeled_idxs)>1:
            try:
                idx=self.activeSearch.next_message
                near_dupes=[]
                if self.dedupe
                    currhash = self.hashed[self.activeSearch.next_message]
                    near_dupes=self.hashlookup[currhash]
                self.activeSearch.interestingMessage()
                self.unlabeled_idxs.remove(idx)
                for idx in near_dupes[1:]:
                    if x in self.unlabeled_idxs:
                        self.activeSearch.setLabel(x,1)
                        self.unlabeled_idxs.remove(x)

            except,e:
                print e
        elif len(self.activeSearch.unlabeled_idxs)==1:
            idx=self.activeSearch.next_message
            self.activeSearch.labels[idx] = 0
            self.activeSearch.unlabeled_idxs.remove(idx)
            self.unlabeled_idxs.remove(idx)

    
    def boringMessage(self):
        if len(self.activeSearch.unlabeled_idxs)>1:
            try:
                idx=self.activeSearch.next_message
                near_dupes=[]
                if self.dedupe
                    currhash = self.hashed[self.activeSearch.next_message]
                    near_dupes=self.hashlookup[currhash]
                self.activeSearch.boringMessage()
                self.unlabeled_idxs.remove(idx)
                for idx in near_dupes[1:]:
                    if x in self.unlabeled_idxs:
                        self.activeSearch.setLabel(x,0)
                        self.unlabeled_idxs.remove(x)

            except,e:
                print e
        elif len(self.activeSearch.unlabeled_idxs)==1:
            idx=self.activeSearch.next_message
            self.activeSearch.labels[idx] = 0
            self.activeSearch.unlabeled_idxs.remove(idx)
            self.unlabeled_idxs.remove(idx)

    def setLabel(self,ext_id,lbl):
        #map from external id to internal corpus idx and then from copus idx to AS matrix idx
        int_idx = self.id_to_idx[ext_id]
        if int_idx in self.unlabeled_idxs:
            if self.dedupe:
                currhash = self.hashed[int_idx]
                near_dupes=self.hashlookup[currhash]
                for x in near_dupes[1:]:
                    if x in self.unlabeled_idxs:
                        self.activeSearch.setLabel(x,lbl)
                        self.unlabeled_idxs.remove(x)
            else:
                self.activeSearch.setLabel(int_idx,lbl)
                self.unlabeled_idxs.remove(int_idx)

    def setLabelBulk(self,ids,lbls):
        for i,ext_id in enumerate(ids):
            self.setLabel(ext_id,lbls[i])
            
    def hashing(self,text):
        n_grams = [x for x in ngrams(text.split(),5)]
        if len (n_grams) > 0:
            return min([int(hashlib.sha256(gram).hexdigest(),16) for gram in n_grams])
        else:
            int(hashlib.sha256("").hexdigest(),16)
        
        
    def getVocabulary(self,text,extendedVoc=[],ngram_range=(1,1),max_df=0.95,min_df=0.005):
        cvec=CountVectorizer(analyzer='word',ngram_range=ngram_range,max_df=max_df,min_df=min_df)
        cvec.fit(text)
        vocab=cvec.vocabulary_
        idx=len(vocab)
        for ngram in extendedVoc:
            if ngram not in vocab:
                vocab[ngram]=idx
                idx+=1
        return vocab
        
        
    def returnWeights(self,unlabeled_only=True,number=20,deduped=True):#return list of (ids,weights)
        if unlabeled_only:
            l = [(self.curr_corpus[idx][0],self.activeSearch.f[idx])for idx in self.activeSearch.unlabeled_idxs]
            l = sorted(l, key=lambda x: x[1],reverse=True)
        else:
            l= sorted([(self.curr_corpus[idx][0],self.activeSearch.f[idx]) for idx in self.activeSearch.f],key=lambda x: x[1],reverse=True)
        if deduped:
            count=0
            toreturn=[]
            s=[]
            for x in l:
                if len(toreturn)<number:
                    hashed=self.hashed[self.id_to_idx[x[0]]]
                    for y in s:
                        if hashed==y:#skip if near duplicate is already in returned set
                            continue
                    #no element in s is a near duplicate of x, so return it
                    s.append(hashed)
                    toreturn.append(x)
                else:
                    return toreturn
            #unlabeled, deduplicated are fewer than number
            return toreturn
        else:
            return l[:number]
                    
        
    def setCountVectorizer(self,vocab=None,binary=True,ngram_range=(1,1),max_df=0.95,min_df=0.005):
        if vocab:
            self.vectorizer=CountVectorizer(analyzer='word',vocabulary=vocab,binary=binary,ngram_range=ngram_range,max_df=max_df,min_df=min_df,decode_error=u'ignore')
        else:
            self.vectorizer=CountVectorizer(analyzer='word',binary=binary,ngram_range=ngram_range,max_df=max_df,min_df=min_df,decode_error=u'ignore')
    
    def setTfidf(self,vocab=None,ngram_range=(1,1),max_df=0.95,min_df=0.005):
        if vocab:
            self.vectorizer=TfidfVectorizer(norm='l1',vocabulary=vocab,stop_words='english',analyzer='word',ngram_range=ngram_range,max_df=max_df,min_df=min_df,decode_error=u'ignore')
        else:
            self.vectorizer=TfidfVectorizer(norm='l1',stop_words='english',analyzer='word',ngram_range=ngram_range,max_df=max_df,min_df=min_df,decode_error=u'ignore')
        

    def get_random_message(self):
        mid=randint(0,self.num_messages)
        return self.curr_corpus[mid][0]
        
    def get_text(self,idx):
        return self.curr_corpus[idx][1]
    
    def get_id(self,idx):
        return self.curr_corpus[idx][0]
        
    def get_next_text(self):
        if len(self.activeSearch.unlabeled_idxs)>0:
            m=self.activeSearch.getNextMessage()
            return self.curr_corpus[m][1]
        else:
            return ""
        
    def get_next(self):
        if len(self.activeSearch.unlabeled_idxs)>0:
            m=self.activeSearch.getNextMessage()
            return self.curr_corpus[m]
        else:
            return (-1,"")
       
    def get_next_id(self):
        if len(self.activeSearch.unlabeled_idxs)>0:
            m=self.activeSearch.getNextMessage()
            return self.curr_corpus[m][0]
        else:
            return -1
        
    def newActiveSearch(self,jsonobj,starting_points,labeled_corpus=[],labels=[],dedupe=False,tfidf=True,dimred=True,n_components=100,prevalence=0.1,lmimdbfeatures=False,eta=0.2):
        #store parameter selections
        #corpus=[(x['_id'],x['_source']['text']) for x in jsonobj]
        corpus=[(x['ad_id'],x['text']) for x in jsonobj]
        self.dedupe=dedupe
        self.tfidf=tfidf
        self.prevalence=prevalence
        self.lmimdbfeatures=lmimdbfeatures
        self.eta=eta
        self.dimred=dimred
        self.n_components=n_components
        self.startAS(corpus,labeled_corpus=labeled_corpus,labels=labels,starting_points=starting_points)
       
    def next_AS(self,jsonobj,starting_points=[]):
        #corpus=[(x['_id'],x['_source']['text']) for x in jsonobj]
        corpus=[(x['ad_id'],x['text']) for x in jsonobj]
        new_labeled_indices=[i+self.start_idx for i,x in enumerate(self.activeSearch.labels[self.start_idx:]) if x !=-1]
        prev_labels=[self.activeSearch.labels[x] for x in new_labeled_indices]#list comprehension
        prev_corpus=[self.curr_corpus[x] for x in new_labeled_indices]
        self.startAS(corpus,labeled_corpus=prev_corpus,labels=prev_labels,starting_points=starting_points)
        
    def startAS(self,corpus,labeled_corpus=[],labels=[],starting_points=[]):
        """
        corpus --> list of touples (id,text) where id is external id
        """
        num_labels = len(labels)
        if num_labels != len(labeled_corpus):
            raise Exception ("Number of lables and number of previously labeled objects does not match")
        if num_labels > 0:
            self.prev_corpus.extend(labeled_corpus)
            self.prev_labels.extend(labels)
            
        #initialise with previous information
        self.start_idx=len(self.prev_labels)    

        #get map from external id to internal index for the new corpus 
        self.id_to_idx={}#maps external id (e.g. AdId) to internal index
        for i,el in enumerate(corpus):
            self.id_to_idx[el[0]]=i+self.start_idx #do not include indices pointing to already labeled objects from previous AS
        self.curr_corpus=corpus
        self.num_messages=len(corpus)
        self.unlabeled_idxs=set(xrange(self.start_idx,self.num_messages))
        self.hashlookup={}
        if self.dedupe:
            #calculate all minhash values
            self.hashed=[self.hashing(tup[1].lower()) for i,tup in enumerate(corpus)]#minhash
            #for now, save collisions in a dictionary. Replace with locality sensitive hashing later
            for i,h in enumerate(self.hashed):
                if h in self.hashlookup:
                    self.hashlookup[h].append(i)
                else:
                    self.hashlookup[h]=[i]
        text = [x[1] for x in self.prev_corpus] + [y[1] for y in corpus]
        
        #save text so that restart is possible
        self.text=text
        #featurize
        ngram_range=(500,0)
        if len(self.extendedVocabulary)==0:
            ngram_range=(1,1)
        for x in self.extendedVocabulary:
            l=len(x.split())
            ngram_range=(min((l,ngram_range[0])),max((l,ngram_range[1])))
        vocabulary = self.getVocabulary(text,extendedVoc=list(self.extendedVocabulary))
        if self.tfidf:
            self.setTfidf(vocab=vocabulary,ngram_range=ngram_range)
        else:
            self.setCountVectorizer(vocab=vocabulary,ngram_range=ngram_range)
        X=self.vectorizer.fit_transform(text)
        #add column to make sure induced graph is fully connected
        self.Xsparse = sparse.hstack((X, sparse.csr_matrix(np.full((X.shape[0],1), X.data.min()*.1 ))))
        #self.X = preprocessing.scale(self.X)
        
        
        if self.dimred:
            print self.Xsparse.shape
            svd=TruncatedSVD(n_components=self.n_components)
            X=svd.fit_transform(self.Xsparse)
            print("dimensionalty reduction leads to explained variance ratio sum of "+str(svd.explained_variance_ratio_.sum()))
            self.sparse=False
        else:
            X=self.Xsparse

        #get scale
        #extimate pairwise distances through random sampling
        pairwise_dists = squareform(pdist(X[np.random.choice(X.shape[0], 1000, replace=False),:], 'euclidean'))
        self.scalefactor = np.mean(pairwise_dists)
        
        params=asI.Parameters(pi=self.prevalence,verbose=False,sparse=self.sparse,eta=self.eta)
        self.activeSearch = asI.kernelAS(params=params) ##fast
        
        self.activeSearch.initialize(X.transpose(),init_labels = {key: value for key, value in enumerate(self.prev_labels)})
        if len(starting_points)==0:
            if len(self.prev_labels)==0:
                raise Exception ("No start point and no labels provided")
        else:
            for x in starting_points:
                self.setLabel(x,1)

        
    def extendAS(self,ext_vocab=[]):
            
        ext_vocab=[x.lower() for x in ext_vocab]
            
        #this is just a restart so labels are still valid
        labels={key: value for key, value in enumerate(self.activeSearch.labels.tolist()) if value > -1}
        self.extendedVocabulary.update(set(ext_vocab))
        
        #attach only 
        ngram_range=(500,0)
        if len(ext_vocab)==0:
            return
        for x in ext_vocab:
            l=len(x.split())
            ngram_range=(min((l,ngram_range[0])),max((l,ngram_range[1])))
        tempvectorizer=CountVectorizer(analyzer='word',vocabulary=ext_vocab,binary=True,ngram_range=ngram_range,decode_error=u'ignore')
        addX=tempvectorizer.fit_transform(self.text)
        #scale by mean distance and some factor
        #some_factor=2
        #addX.multiply(self.scalefactor*float(some_factor))
        
        #add column 
        self.Xsparse = sparse.hstack((self.Xsparse,addX))
        
        if self.dimred:
            print self.Xsparse.shape
            svd=TruncatedSVD(n_components=self.n_components)
            X=svd.fit_transform(self.Xsparse)
            print("dimensionalty reduction leads to explained variance ratio sum of "+str(svd.explained_variance_ratio_.sum()))
            self.sparse=False
        else:
            X=self.Xsparse
        params=asI.Parameters(pi=self.prevalence,verbose=False,sparse=self.sparse,eta=self.eta)
        self.activeSearch = asI.kernelAS(params=params) ##fast
        self.activeSearch.initialize(X.transpose(),init_labels = labels)
        


