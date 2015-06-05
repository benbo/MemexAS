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
