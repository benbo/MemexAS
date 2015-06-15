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
        self.index_map={}#maps active search matrix indices to indices of corpus
        self.index_map_reverse={}#maps indices from corpus to active search matrix indices
        self.near_duplicates={}#for indices in corpus points to indices of near duplicates        
        self.num_messages=-1
        self.start_idx=0
        self.extendedVocabulary=set()
        
        self.scalefactor=0
        self.dedupe=True
        self.tfidf=False
        self.prevalence=0.1
        self.eta=0.5
        self.dimred=False
        self.sparse=True
        #variables needed to clean text


    def interestingMessage(self):
        if len(self.activeSearch.unlabeled_idxs)>1:
            self.activeSearch.interestingMessage()
        elif len(self.activeSearch.unlabeled_idxs)==1:
            idx=self.activeSearch.next_message
            self.activeSearch.labels[idx] = 1
            self.activeSearch.unlabeled_idxs.remove(idx)
    
    def boringMessage(self):
        if len(self.activeSearch.unlabeled_idxs)>1:
            self.activeSearch.boringMessage()
        elif len(self.activeSearch.unlabeled_idxs)==1:
            idx=self.activeSearch.next_message
            self.activeSearch.labels[idx] = 0
            self.activeSearch.unlabeled_idxs.remove(idx)

    def setLabel(self,idx,lbl):
        #map from external id to internal corpus idx and then from copus idx to AS matrix idx
        int_idx = self.index_map_reverse[self.id_to_idx[idx]]
        if int_idx in self.activeSearch.unlabeled_idxs:
            self.activeSearch.setLabel(int_idx,lbl)

    def setLabelBulk(self,ids,lbls):
        lookup=frozenset(self.activeSearch.unlabeled_idxs)
        for i,tid in enumerate(ids):
            #map from external id to internal corpus idx and then from copus idx to AS matrix idx
            int_idx = self.index_map_reverse[self.id_to_idx[tid]]
            if int_idx in lookup:
                self.activeSearch.setLabel(int_idx,lbls[i])
    
            
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
            l = [(self.curr_corpus[self.index_map[idx]],self.activeSearch.f[idx])for idx in self.activeSearch.unlabeled_idxs]
            l = sorted(l, key=lambda x: x[1],reverse=True)
        else:
            weights=[0.0]*self.num_messages
            for key in self.index_map:
                for idx in self.near_duplicates[self.index_map[key]]:
                    weights[idx]=self.activeSearch.f[idx]
            l= sorted([(self.curr_corpus[i][0],w) for i,w in enumerate(weights)],key=lambda x: x[1],reverse=True)
        if deduped:
            count=0
            toreturn=[]
            s=[]
            for x in l:
                if len(toreturn)<number:
                    hashed=self.hashing(x[0][1].lower())
                    for y in s:
                        if hashed==y:#skip if near duplicate is already in returned set
                            continue
                    #no element in s is a near duplicate of x, so return it
                    s.append(hashed)
                    toreturn.append((x[0][0],x[1]))
                else:
                    return toreturn
            #unlabeled, deduplicated are fewer than number
            return toreturn
        else:
            return [(x[0][0],x[1]) for x in l[:number]]
                    
        
    def setCountVectorizer(self,vocab=None,binary=True,ngram_range=(1,1),max_df=0.95,min_df=0.005):
        if vocab:
            self.vectorizer=CountVectorizer(analyzer='word',vocabulary=vocab,binary=binary,ngram_range=ngram_range,max_df=max_df,min_df=min_df)
        else:
            self.vectorizer=CountVectorizer(analyzer='word',binary=binary,ngram_range=ngram_range,max_df=max_df,min_df=min_df)
    
    def setTfidf(self,vocab=None,ngram_range=(1,1),max_df=0.95,min_df=0.005):
        if vocab:
            self.vectorizer=TfidfVectorizer(norm='l1',vocabulary=vocab,stop_words='english',analyzer='word',ngram_range=ngram_range,max_df=max_df,min_df=min_df)
        else:
            self.vectorizer=TfidfVectorizer(norm='l1',stop_words='english',analyzer='word',ngram_range=ngram_range,max_df=max_df,min_df=min_df)
        

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
            return self.curr_corpus[self.index_map[m]][1]
        else:
            return ""
        
    def get_next(self):
        if len(self.activeSearch.unlabeled_idxs)>0:
            m=self.activeSearch.getNextMessage()
            return self.curr_corpus[self.index_map[m]]
        else:
            return (-1,"")
       
    def get_next_id(self):
        if len(self.activeSearch.unlabeled_idxs)>0:
            m=self.activeSearch.getNextMessage()
            return self.curr_corpus[self.index_map[m]][0]
        else:
            return -1
        
    def decide_startingpoint(self,id):
        self.activeSearch.firstMessage(self.index_map_reverse[self.id_to_idx[id]])        
        
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
        prev_corpus=[self.curr_corpus[self.index_map[x]] for x in new_labeled_indices]
        self.startAS(corpus,labeled_corpus=prev_corpus,labels=prev_labels,starting_points=starting_points)
        
        
    def deduplicate(self,corpus):
        # reduce size of corpus passed on to AS by removing near-duplicates
        # create a mapping from original corpus passed to function to the new, smaller alt_copus 
        # where near-duplicates have been removed
        alt_corpus=[]
        flag=True

        hashed=[self.hashing(tup[1].lower()) for i,tup in enumerate(corpus)]#minhash
        count=self.start_idx#initialize with number of previously labeled objects
        for i,x in enumerate(hashed):
            for y in alt_corpus:
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
        return [corpus[y[0]][1] for y in alt_corpus]
        
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
        self.index_map={}#maps active search matrix indices to indices of curr_corpus
        self.index_map_reverse={}#maps indices from corpus to active search matrix indices
        self.near_duplicates={}#for indices in corpus points to indices of near duplicates 
        

        text=[]
        if self.dedupe:
            text = [x[1] for x in self.prev_corpus] + self.deduplicate(corpus)
        else:
            count=self.start_idx
            for i in range(self.num_messages):
                self.index_map[count]=i
                self.index_map_reverse[i]=count
                self.near_duplicates[i]=[i]
                count+=1           
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
        self.X=self.vectorizer.fit_transform(text)
        #add column to make sure induced graph is fully connected
        self.X = sparse.hstack((self.X, sparse.csr_matrix(np.full((self.X.shape[0],1), self.X.data.min()*.1 ))))
        #self.X = preprocessing.scale(self.X)
        
        
        if self.dimred:
            print self.X.shape
            svd=TruncatedSVD(n_components=self.n_components)
            self.X=svd.fit_transform(self.X)
            print("dimensionalty reduction leads to explained variance ratio sum of "+str(svd.explained_variance_ratio_.sum()))
            self.sparse=False

        
        params=asI.Parameters(pi=self.prevalence,verbose=False,sparse=self.sparse,eta=self.eta)
        self.activeSearch = asI.kernelAS(params=params) ##fast
        
        self.activeSearch.initialize(self.X.transpose(),init_labels = {key: value for key, value in enumerate(self.prev_labels)})
        if len(starting_points)==0:
            if len(self.prev_labels)==0:
                raise Exception ("No start point and no labels provided")
        else:
            self.decide_startingpoint(starting_points[0])
            for x in starting_points[1:]:
                self.activeSearch.setLabel(self.index_map_reverse[self.id_to_idx[i]],1)

        
    def extendAS(self,ext_vocab=[]):

            
        ext_vocab=[x.lower() for x in ext_vocab]
            
        #this is just a restart so labels are still valid
        labels={key: value for key, value in enumerate(self.activeSearch.labels.tolist()) if value > -1}
        self.extendedVocabulary.update(set(ext_vocab))
        
        #redo everything
        ngram_range=(500,0)
        if len(ext_vocab)==0:
            return
        for x in self.extendedVocabulary:
            l=len(x.split())
            ngram_range=(min((l,ngram_range[0])),max((l,ngram_range[1])))
        
        
        vocabulary = self.getVocabulary(self.text,extendedVoc=list(self.extendedVocabulary))
        if self.tfidf:
            self.setTfidf(vocab=vocabulary,ngram_range=ngram_range)
        else:
            self.setCountVectorizer(vocab=vocabulary,ngram_range=ngram_range)
        self.X=self.vectorizer.fit_transform(self.text)
        #add column to make sure induced graph is fully connected
        self.X = sparse.hstack((self.X, sparse.csr_matrix(np.full((self.X.shape[0],1), self.X.data.min()*.1 ))))

        #if self.dimred:
        #    print self.X.shape
        #    svd=TruncatedSVD(n_components=self.n_components)
        #    self.X=svd.fit_transform(self.X)
        #    print("dimensionalty reduction leads to explained variance ratio sum of "+str(svd.explained_variance_ratio_.sum()))
        #    self.sparse=False
        
        #attach only 
        #ngram_range=(500,0)
        #if len(ext_vocab)==0:
        #    return
        #for x in ext_vocab:
        #    l=len(x.split())
        #    ngram_range=(min((l,ngram_range[0])),max((l,ngram_range[1])))
        #tempvectorizer=CountVectorizer(analyzer='word',vocabulary=ext_vocab,binary=True,ngram_range=ngram_range)
        #addX=tempvectorizer.fit_transform(text)
        #add column 
        #self.X = sparse.hstack((self.X,addX))
        #get scale
        #pairwise_dists = squareform(pdist([np.random.choice(self.X.shape[0], 1000, replace=False),:], 'euclidean'))
        #self.scalefactor = np.median(pairwise_dists)
        

        
        params=asI.Parameters(pi=self.prevalence,verbose=False,sparse=self.sparse,eta=self.eta)
        self.activeSearch = asI.kernelAS(params=params) ##fast
        self.activeSearch.initialize(self.X.transpose(),init_labels = labels)
        


