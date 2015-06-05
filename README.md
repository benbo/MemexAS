# MemexAS
Active Search (AS) wrapper to conduct AS on deduplicated texts and transfer labels unto a new corpus

##Dependencies##
- numpy
- scipy 
- cython
- scikit-learn

##Installation##
wget https://github.com/benbo/MemexAS/archive/master.zip  
unzip master.zip  
mv MemexAS-master MemexAS  
cd MemexAS  
\#build a fast ngram cython function  
python setup.py build_ext --inplace  
\#Additionally, you need to download the following file into the same directory 
wget https://github.com/AutonlabCMU/ActiveSearch/blob/sibi/kernelAS/python/activeSearchInterface.py
##Example##
If you have followed those steps then you should be able to use the MemexAS class in MemexAS.py
    from os import listdir  
    from os.path import join  
    import MemexAS  
    myAS=MemexAS()  
    mypath="mytext.txt"#path to a textfile where each line represents a document/Ad  
    text=[(i,line.rstrip()) for i,line in enumerate(open(join(mypath,f),'r'))]  
    myAS.newActiveSearch(text,starting_points=[startp])  


