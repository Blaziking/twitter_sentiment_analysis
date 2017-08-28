import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize



class VoteClassifier(ClassifierI):
	def __init__(self, *classifiers):
		self.classifiers = classifiers

	def classify(self, features):
		votes = []
		for c in self.classifiers:
			v = c.classify(features)
			votes.append(v)
		return mode(votes)

	def confidence(self, features):
		votes = []
		for c in self.classifiers:
			v = c.classify(features)
			votes.append(v)

		choice_votes = votes.count(mode(votes))
		conf = choice_votes / len(votes)
		return conf

# short_pos =open("short_reviews/positive.txt", "r").read() #doubt : why r read both used ?
# short_neg =open("short_reviews/negative.txt", "r").read()

documents = []


all_words = []

# allowed_words = ['J']

# for r in short_pos.split('\n'):
# 	documents.append((r, "pos"))
# 	words = word_tokenize(r)
# 	pos = nltk.pos_tag(words)
# 	for w in pos: 
# 		if w[1][0] in allowed_words:
# 			all_words.append(w[0].lower())


# for r in short_neg.split('\n'):
# 	documents.append((r, "neg"))
# 	words = word_tokenize(r)
# 	pos = nltk.pos_tag(words)
# 	for w in pos: 
# 		if w[1][0] in allowed_words:
# 			all_words.append(w[0].lower())







#pickle for documents

# save_documents = open('twitter/documents.pickle', 'wb')
# pickle.dump(documents, save_documents)
# save_documents.close()

documents_f = open("twitter\documents.pickle", "rb")
documents = pickle.load(documents_f)
documents_f.close()


#PICKLE FOR ALL WORDS

# save_all_words = open('twitter/all_words.pickle', 'wb')
# pickle.dump(all_words, save_all_words)
# save_all_words.close()

allwords_f = open("twitter/all_words.pickle","rb")
all_words = pickle.load(allwords_f)
allwords_f.close()



all_words = nltk.FreqDist(all_words)
word_features = list(all_words.keys())[:5000]


def find_features(document):
	words = word_tokenize(document)
	features = {}
	for w in word_features:
		features[w] = (w in words)

	return features







featuresets = [(find_features(rev), category) for (rev, category) in documents]


# save_featuresets = open('twitter/featuresets.pickle', 'wb')
# pickle.dump(featuresets, save_featuresets)
# save_featuresets.close()

# classifier_f = open("twitter/featuresets.pickle", "rb")
# featuresets = pickle.load(classifier_f)
# classifier_f.close()



# random.shuffle(featuresets)

# training_set  = featuresets[:10000]
# testing_set = featuresets[10000:]


classifier_f = open("twitter/naivebayes.pickle", "rb")
classifier = pickle.load(classifier_f)
classifier_f.close()


# classifier = nltk.NaiveBayesClassifier.train(training_set)
# print("Original Naive Bayes Algo Accuracy percent:", (nltk.classify.accuracy(classifier, testing_set))*100)
# save_classifier = open("twitter/naivebayes.pickle", "wb")
# pickle.dump(classifier, save_classifier)
# save_classifier.close()





classifier_f = open("twitter/MNB_classifier.pickle", "rb")
MNB_classifier = pickle.load(classifier_f)
classifier_f.close()
# MNB_classifier = SklearnClassifier(MultinomialNB())
# MNB_classifier.train(training_set)
# print("MNB Naive Bayes Algo Accuracy percent:", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)


# save_classifier = open('twitter/MNB_classifier.pickle', 'wb')
# pickle.dump(MNB_classifier, save_classifier)
# save_classifier.close()





classifier_f = open("twitter/LogisticRegression.pickle", "rb")
LogisticRegression_classifier = pickle.load(classifier_f)
classifier_f.close()
# LogisticRegression_classifier = SklearnClassifier(BernoulliNB())
# LogisticRegression_classifier.train(training_set)
# print("LogisticRegression Naive Bayes Algo Accuracy percent:", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)
# save_classifier = open("twitter/LogisticRegression.pickle", "wb")
# pickle.dump(LogisticRegression_classifier, save_classifier)
# save_classifier.close()







classifier_f = open("twitter/Bernoulli.pickle", "rb")
Bernoulli_classifier = pickle.load(classifier_f)
classifier_f.close()

# Bernoulli_classifier = SklearnClassifier(BernoulliNB())
# Bernoulli_classifier.train(training_set)
# print("Bernoulli Naive Bayes Algo Accuracy percent:", (nltk.classify.accuracy(Bernoulli_classifier, testing_set))*100)
# save_classifier = open("twitter/Bernoulli.pickle", "wb")
# pickle.dump(Bernoulli_classifier, save_classifier)
# save_classifier.close()











classifier_f = open("twitter/SGDC.pickle", "rb")
SGDC_classifier = pickle.load(classifier_f)
classifier_f.close()

# SGDC_classifier = SklearnClassifier(SGDClassifier())
# SGDC_classifier.train(training_set)
# print("SGDC Naive Bayes Algo Accuracy percent:", (nltk.classify.accuracy(SGDC_classifier, testing_set))*100)
# save_classifier = open("twitter/SGDC.pickle", "wb")
# pickle.dump(SGDC_classifier, save_classifier)
# save_classifier.close()
















classifier_f = open("twitter/LinearSVC.pickle", "rb")
LinearSVC_classifier = pickle.load(classifier_f)
classifier_f.close()
# LinearSVC_classifier = SklearnClassifier(LinearSVC())
# LinearSVC_classifier.train(training_set)
# print("LinearSVC Naive Bayes Algo Accuracy percent:", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)
# save_classifier = open("twitter/LinearSVC.pickle", "wb")
# pickle.dump(LinearSVC_classifier, save_classifier)
# save_classifier.close()






# classifier_f = open("twitter/NuSVC.pickle", "rb")
# NuSVC_classifier = pickle.load(classifier_f)
# classifier_f.close()
# NuSVC_classifier = SklearnClassifier(NuSVC())
# NuSVC_classifier.train(training_set)
# print("NuSVC Naive Bayes Algo Accuracy percent:", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)
# save_classifier = open("twitter/NuSVC.pickle", "wb")
# pickle.dump(NuSVC_classifier, save_classifier)
# save_classifier.close()



voted_classifier = VoteClassifier(classifier,MNB_classifier,LinearSVC_classifier,LogisticRegression_classifier,SGDC_classifier)

# print("voted_classifier Algo Accuracy percent:", (nltk.classify.accuracy(voted_classifier, testing_set))*100)

def sentiment(text):
	feats = find_features(text)
	return voted_classifier.classify(feats), voted_classifier.confidence(feats)




