### MACIEJ SKORSKI ###

# this scripts demonstrates the use of the bag-of-words model to text classification

# problem: classify text documents (tech-related web articles); data format: [class id, article title, article content]

# data: attached file data.csv

# data representation for ML: we use the bag-of-words model, counting words occurencies within documents (standard technique)

# feature selection: 
# a) in the first attempt we try inverse document frequencies (which prefers less common - more informative - words); 
# b) as the second solution we try entropy-based word selection (consider only words with high information w.r.t classes); computationally expensive !

# algorithms: we play with two algorthms widely used for text classification
# a) Naive Bayes
# b) Support Vector Classifiers

# implementation details: 
# a) we use class LinearSVC which optimizes SCV with linear kernels
# b) we use sparse matrices when computing frequencies 

# navigation

"""

DATA EXTRACTION

DATA CLEANSING

WARMUP MODEL (SIMPLE BAG-OF-WORDS + NAIVE BAYES + 5-FOLD CROSS VALIDATION)

FURTHER IMPROVEMENTS

	- PIPLELINE	

	- RARE CLASSES

	- FILTER STOP-WORDS

	- BI-GRAM WORDS

	- ANALYZE FULL TEXTS

	- WORDS PRESELECTION (ENTROPY BASED)

"""




import os
import pandas as pnd
import numpy as np
import sklearn
import math

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline

from scipy.stats import entropy


# data root dir

data_dir = '/home/mskorski/Documents/Crypto/MachineLearning/web_NLP/'

os.chdir(data_dir)

### DATA EXTRACTION ###

# assuming the data fits in memory
data = pnd.read_csv('techmeme-5k.csv', header=0)
data.shape

 # structure: article_id, cluster_id, title, content

### DATA CLEANSING ###

def preprocess(dt):
	# ignore article ids
	dt = dt.drop('aid',axis=1)
	# map the clusters ids to numbers
	cids = data['cid'].unique()
	cids_range = len(cids) 
	cids_dict = {k:v for k,v in zip(cids,range(len(cids)))}
	dt['cid'] = dt['cid'].apply(lambda t: cids_dict[t])
	return dt

data = preprocess(data)

# class distribituion: 482 clusters in the traning data, but some of them are rare, too small to learn !!!
from matplotlib import pyplot as plt
data['cid'].plot.hist(200)
plt.show()

# we may want to smooth the class distribituion by ignoring negligible classes (the threshold is a hyperparameter) !!!
min_class_size = 5
class_counts = pnd.crosstab(data['cid'],1)
print("threshold=%s class number=%s data coverage=%s" % (min_class_size, np.float((class_counts>=min_class_size).sum()), np.float( class_counts[class_counts>=min_class_size].sum()/len(data) )) )
# threshold = 10, ~200 clusters, ~75% of the data

def smooth_rare_classes(dt,min_class_size):
	class_counts = pnd.crosstab(dt['cid'],1)
	good_classes = np.array(class_counts.index[class_counts[1]>=min_class_size])
	return dt[ dt['cid'].isin(good_classes) ]

X_train = data
y_train = X_train['cid']
y_train = X_train['cid']
X_train = X_train.drop('cid',axis=1)

### WARMUP MODEL - SIMPLE BAG-OF-WORDS + NAIVE BAYES + 5-FOLD CROSS VALIDATION ###

## FEATURE SELECTION AND REPRESENTATION ##

# bag-of-words model: words extracted from all titles, features: inverse document frequencies (prefers less common words)
# term frequencies are sparse at the collection level (good terms are class-speficic); therefore sparse memory structure are used (implemented in CountVectorizer)

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
# term-document matrix
X_train_f = count_vect.fit_transform(X_train['title'])
# note the matrix is really sparse
print("sparse compression = %.4f" % ((X_train_f>0).sum()/( X_train_f.shape[0]*X_train_f.shape[1] )))

from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer().fit(X_train_f)
# reweight frequencies
X_train_tfidf = tfidf_transformer.transform(X_train_f)

## TRAIN THE MODEL ##

from sklearn.naive_bayes import MultinomialNB
# MNB classifier with Laplace smoothing term (avoids zero divisions when calculating empirical distributions)
clf = MultinomialNB(alpha=1).fit(X_train_tfidf, y_train)

# print sample features - most informative words in 20 first clusters 
def print_features(count_vect,clf):
	features = np.array(count_vect.get_feature_names())
	informative_words = []
	for i,c in enumerate(clf.classes_):
		top10 = np.argsort(clf.coef_[i])[-10:]
		informative_words.append("%s" % (",".join(features[top10])))
	for i in range(7):
		print("%s:%s" % (i,informative_words[i]))
print_features(count_vect,clf)

## TEST THE MODEL ##

# evaluation by 5-fold cross-validation; note we don't have enough data to extract on a hand-out testing set (too many classes)

def report(clf,X_train,y_train):
	results = cross_val_score(clf,X_train,y_train,cv=5)
	print("mean {0} +/- 2* {1}".format(results.mean(),results.std()))

report(clf,X_train_tfidf,y_train)
# average accuracy ~42%, note the high variance!

### FURTHER IMPROVEMENTS ###

# in this section we improve the solution, using the constructed Bayes model as a benchmark

## PIPLELINE ##

text_clf = Pipeline([
			('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', MultinomialNB()),
])

## RARE CLASSES ##

# note that the classifier performs much better on more frequent classes

X_test = smooth_rare_classes(data,10)
y_test = X_test['cid']
report(text_clf,X_test['title'],y_test)
# accuracy ~ 58%

# in fact, some classes are too small to learn and validate results
X_test = smooth_rare_classes(data,0)
y_test = X_test['cid']
report(text_clf,X_test['title'],y_test)

## FILTER STOP-WORDS ##

text_clf = Pipeline([
			('vect', CountVectorizer(stop_words = 'english')),
            ('tfidf', TfidfTransformer()),
            ('clf', MultinomialNB()),
])
report(text_clf,X_train['title'],y_train)
# accuracy improved to ~ 48% on average :-)
text_clf.fit(X_train['title'],y_train)
print_features(text_clf.steps[0][1],text_clf.steps[2][1])
# indeed, stop-words like off disappeared (see the first two lines)

## BI-GRAM WORDS ##

# our bag-of-words is built from single words, pairs of consecutive words (bigrams) may be more accurate 
text_clf = Pipeline([
			('vect', CountVectorizer(ngram_range=(1,2))),
            ('tfidf', TfidfTransformer()),
            ('clf', MultinomialNB()),
])
report(text_clf,X_train['title'],y_train)
# didn't help :-(

## ANALYZE FULL TEXTS ##

# try to analyze only texts
text_clf = Pipeline([
			('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', MultinomialNB()),
])
report(text_clf,X_train['text'],y_train)
# didn't help :-( 

# merge title and text data
X_train0 = X_train.apply(lambda t: t['title']+'.'+t['text'],axis=1)
text_clf = Pipeline([
			('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', MultinomialNB())
])
report(text_clf,X_train0,y_train)
# didn't help either :-( maybe because full texts are too noisy 

## WORDS PRESELECTION (ENTROPY BASED) ##

# select vocabulary by fancy information-theoretic methods (entropy, divergences), instead of all words
# for good results we need probably less than 10 key words per class as in the Reuters-RCV1 problem (https://nlp.stanford.edu/IR-book/html/htmledition/mutual-information-1.html#fig:mccallum)

# idea: minimum entropy principle - most informative words are those which minimize entropy of a class: ShannonEntropy(class | term)
# complexity: entropy computations are expensive, the code below run for ~ 1hr
# validation: can't afford computations for cv folds, therefore we evaluate accuracy on a random train/test split (note: this is a rough estimate because of rare classes!)

X_train0, X_test0, y_train0, y_test0 = train_test_split(X_train,y_train,train_size=0.7,test_size=0.3, random_state=11)

# note: need to repeat this for every cv fold, or test with a handout set

def best_entropy_terms(X_train,y_train):
	""" for every class in y_train, outputs most informative features for documents in X_train """
	words_per_class = 40
	entropy_terms = {}
	percentage = 0
	step = len(y_train.unique())*0.05
	i = 0
	# build document-term matrix
	vect = CountVectorizer()
	term_matrix = vect.fit_transform(X_train['title'])>0
	# reindex 
	y_train = y_train.reset_index(drop=True)
	# compute information contributed to class c by term t
	for c in y_train.unique():
		entr = {}
		tmp0 = term_matrix[list(y_train[y_train == c].index)]
		tmp1 = term_matrix[list(y_train[y_train != c].index)]
		for t in range(term_matrix.shape[1]):
			n11 = tmp0[:,t].sum() 
			n10 = tmp0.shape[0]-n11
			n01 = tmp1[:,t].sum()
			n00 = tmp1.shape[0]-n01
			m1 = np.array([[n00,n01],[n10,n11]])
			m1 = m1.T/m1.sum()
			# conditional entropy formula
			entr[t] = m1[0].sum() * entropy(m1[0]/m1[0].sum()) + m1[1].sum() * entropy(m1[1]/m1[1].sum()) 
		# select 20 terms which minimize entropy
		l = sorted(entr,key=entr.__getitem__)[:words_per_class]
		entropy_terms[c] = ",".join(np.array(vect.get_feature_names())[l])
		i = i+1
		if i > step:
			percentage = percentage + 5
			print("finished %s %%" % percentage)
			i = 0
	# return the array of best features
	return entropy_terms

def write_entropy_terms(entropy_words):
	""" write the vocabulary to a file """
	with open('entropy_words.csv','w') as f:
		for i in entropy_words:
			f.write("%s:%s\n"%(i,entropy_words[i]))

def load_entropy_terms():
	""" use this to load the precomputed vocabulary """
	entropy_words = {}
	d = pnd.read_csv('entropy_words.csv',delimiter = ':', header = None)
	for i in range(len(d[0])):
		entropy_words[d[0][i]] = d[1][i]
	return entropy_words

# compute informative terms by entropy; use the precomputed list (second line below) if it takes too long
entropy_words = best_entropy_terms(X_train0,y_train0)
# entropy_words = load_entropy_terms() - uncomment !

def print_entropy_features():
	for i in range(7):
		print(str(i)+":"+",".join(entropy_words[i].split(',')[:10])) 

print_entropy_features()
# note the words are similar to those found by inverse frequencies

def entropy_method_report(base_clf = MultinomialNB()):
	accuracy = []
	accuracy1 = []
	class_words_n = list(range(1,41))
	for words_per_class in class_words_n:
		vocabulary = []
		for i in entropy_words:
			for j in entropy_words[i].split(',')[:words_per_class]:
				vocabulary.append(j)
		vocabulary = np.unique(vocabulary)
		# without stop words
		text_clf = Pipeline([
					('vect', CountVectorizer(vocabulary=vocabulary)),
		            ('tfidf', TfidfTransformer()),
		            ('clf', base_clf),
		])
		text_clf.fit(X_train0['title'],y_train0)
		accuracy.append( text_clf.score(X_test0['title'],y_test0) )
		# with stop words
		text_clf = Pipeline([
					('vect', CountVectorizer(stop_words = 'english', vocabulary=vocabulary)),
		            ('tfidf', TfidfTransformer()),
		            ('clf', base_clf),
		])
		text_clf.fit(X_train0['title'],y_train0)
		accuracy1.append( text_clf.score(X_test0['title'],y_test0) )

	tmp = pnd.DataFrame([class_words_n, accuracy]).T
	tmp1 = pnd.DataFrame([class_words_n, accuracy1]).T
	plt.plot(tmp[0],tmp[1],'b',label='no stop words')
	plt.plot(tmp1[0],tmp1[1],'r',label='english stop words')
	plt.show()

entropy_method_report()
# accuracy ~ 49% with only few words per class, better than with the full (automatic) vocabulary :-)

## BETTER CLASSIFIER - SVM ##

# we change the algorithm to support vector classifiers, widely regognized as one of the best techniques for text classification 

entropy_method_report(base_clf = LinearSVC())

## TUNNING HYPERPARAMETERS ##

text_clf = Pipeline([
			('vect', CountVectorizer()),
            ('tfidf', TfidfTransformer()),
            ('clf', MultinomialNB()),
])

from sklearn.model_selection import GridSearchCV
parameters = {
	'tfidf__use_idf': (True, False),
	'clf__alpha': (1e-1, 1e-3),
	'vect__ngram_range': ((1,1),(1,2)),
	'vect__stop_words': ('english',None)
	}

gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1, cv=5)
gs_clf = gs_clf.fit(X_train['text'], y_train)

gs_clf.best_params_


report(text_clf,X_train['text'],y_train)

### HANDOUT SET EVALUATION ###





# see https://medium.com/towards-data-science/machine-learning-nlp-text-classification-using-scikit-learn-python-and-nltk-c52b92a7c73a


# mind the inbalance!

# pass the vocabulary when evaluating on fresh data !









