from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import svm

import numpy as np

import eli5
from eli5.lime import TextExplainer

from sklearn.pipeline import Pipeline, make_pipeline


corpus = [
		"The cat (Felis catus) is a small carnivorous mammal.",
		"It is the only domesticated species in the family Felidae and often referred to as the domestic cat to distinguish it from wild members of the family.",
		"The cat is either a house cat, a farm cat or a feral cat",
		"Domestic cats are valued by humans for companionship and for their ability to hunt rodents.", 
		"About cat breeds are recognized by various cat registries.",
		
		"The domestic dog (Canis lupus familiaris when considered a subspecies of the wolf",
		"or Canis familiaris dog when considered a distinct species) is a member of the genus Canis (canines)",
		"which forms part of dog the wolf-like canids, and is the most widely abundant terrestrial carnivore.",
		"The dog and the extant dog gray wolf are sister taxa as modern wolves are not closely related to the wolves",
		"that were first domesticated, dog which implies that the direct ancestor of the dog is extinct. "]
Y = [0,0,0,0,0,1,1,1,1,1]

X_p = ["The dog and the cat are unalike because the dog is"]
Y_p = [1]

#tfidf = TfidfVectorizer(sublinear_tf=True)
tfidf = CountVectorizer()
X = tfidf.fit_transform(corpus).toarray()
Xmat = tfidf.fit_transform(corpus)


feature_names_alpha = tfidf.get_feature_names()
feature_names = tfidf.vocabulary_




print(X.shape, len(feature_names_alpha))

for j in range(X.shape[1]):
	print(j, feature_names_alpha[j], [X[i][j] for i in range(X.shape[0])])

clf = svm.SVC(tol=1e-4, probability=True, 
	kernel='linear', decision_function_shape='ovr' ).fit(Xmat, Y)

pipe = make_pipeline(tfidf, clf)

te = TextExplainer(random_state=42)
te.fit(corpus[0], pipe.predict_proba)
te.show_prediction(target_names=[0,1])

'''
importance = clf.coef_.toarray()[0]

array = ['']*len(feature_names)

for k in feature_names:
	array[feature_names[k]] = k

importance, feature_names = zip(*sorted(zip(importance,feature_names)))
importance, feature_names = np.array(importance), np.array(feature_names)

importance = importance[::-1]
feature_names = feature_names[::-1]

for fn, imp in zip(feature_names, importance):
	print(fn, imp)

print(clf.support_vectors_)
print(clf.support_)
print(clf.n_support_)
'''

'''	
importance = clf.coef_.toarray()
feature_names = tfidf.vocabulary_
sortedt_names = tfidf.get_feature_names()

print(feature_names)
print(len(importance[0]))

for k in sortedt_names:
	print(k, importance[0][feature_names[k]])#, clf.coef_[1][feature_names[k]])
'''