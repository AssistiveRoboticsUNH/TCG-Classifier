from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import svm, metrics

import numpy as np

import eli5
from eli5.lime import TextExplainer
from eli5.sklearn import PermutationImportance

from sklearn.pipeline import Pipeline, make_pipeline

# Setup training/eval data
Xtrain = [
		"The cat (Felis catus) is a small carnivorous mammal.",
		"It is the only domesticated species in the family Felidae and often referred to as the domestic cat to distinguish it from wild members of the family.",
		"The cat is either a house cat, a farm cat or a feral cat",
		"Domestic cats are valued by humans for companionship and for their ability to hunt rodents.", 
		"About cat breeds are recognized by anatomy various cat registries.",
		
		"The domestic dog (Canis lupus familiaris when considered a subspecies of the wolf",
		"or Canis familiaris dog when considered a distinct species) is a member of the genus Canis (canines)",
		"which forms part cat cat of dog the wolf-like canids, and is the most widely abundant terrestrial carnivore.",
		"The dog and the extant dog gray wolf are sister taxa as modern wolves are not closely related to the wolves",
		"that were first domesticated, dog which implies that the direct ancestor of the dog is extinct. ",

		"Rats rat are various medium-sized, long-tailed rodents.",
		"Species of rat rats are found throughout the order Rodentia,",
		"but stereotypical rats are rat found in the genus Rattus. ",
		"Other rat genera rat include Neotoma (pack rats),",
		"Bandicota (bandicoot rats) and rat Dipodomys (kangaroo rats).",
		]


Ytrain = [0,0,0,0,0,1,1,1,1,1,2,2,2,2,2]

Xeval = [
		"The cat is similar in anatomy to the other felid species, ",
		"has a strong flexible body, quick reflexes, sharp teeth and retractable claws cat adapted to killing small prey.",
		"Its night vision and sense of smell are well developed. Cat communication cat includes vocalizations like meowing, purring,",
		"trilling, hissing, growling and grunting as well as cat-specific body cat language. It is a solitary hunter, but a social species.",

		"Their long association with humans has led dogs to be uniquely dog attuned to human behavior",
		"and they are able to thrive on a starch-rich diet that would dog be inadequate for other canids.",
		"Dogs vary widely in shape, size and colors. They perform dog many roles for humans, such as hunting,",
		"herding, pulling loads, protection, assisting police dog and military, companionship and, more recently, aiding disabled people and therapeutic roles.",
		
		"Rats are typically distinguished rat from mice by their size." ,
		"Generally, when someone discovers a rat large muroid rodent, ",
		"its common name includes the term rat, while if it is smaller, ",
		"its name includes the term mouse. The rat common terms rat and mouse are not taxonomically specific.",
		]

Yeval = [0,0,0,0,1,1,1,1,2,2,2,2]


# define model
#tfidf = CountVectorizer()
tfidf = TfidfVectorizer(ngram_range=(1,1), sublinear_tf=True)
clf = svm.SVC(max_iter=100, tol=1e-4, probability=True, 
	kernel='linear', decision_function_shape='ovr' )

Xtrain_tfidf = tfidf.fit_transform(Xtrain)
clf.fit(Xtrain_tfidf, Ytrain)

# fit model
pred = clf.predict(tfidf.transform(Xeval))
print(metrics.accuracy_score(pred, Yeval))

# understanding features
fn = np.array(tfidf.get_feature_names())
print(fn)

coef = clf.coef_.toarray()#.reshape(-1)
print(coef.shape)

for label in range(3):

	print('')
	print("Label: ", label)

	cf = coef[label].reshape(-1)

	order = cf.argsort()
	cf = cf[order][::-1]
	fn = fn[order][::-1]

	for f, c in zip(fn, cf):
		print(f, c)



perm = PermutationImportance(clf).fit(tfidf.transform(Xeval).toarray(), Yeval)
out = eli5.show_weights(perm, feature_names=fn)

print(out.data)


'''
# build LIME TextExplainer
te = TextExplainer(random_state=42)
pipe = make_pipeline(tfidf, clf)
te.fit(Xeval[0], pipe.predict_proba)
#out = te.show_prediction(target_names=[0,1])
out = te.show_weights(target_names=[0,1])#eli5.show_weights(te, feature_names=tfidf.get_feature_names())

print(out.data)
'''









'''
feature_names_alpha = tfidf.get_feature_names()
feature_names = tfidf.vocabulary_

for j in range(X.shape[1]):
	print(j, feature_names_alpha[j], [X[i][j] for i in range(X.shape[0])])
'''
'''
clf = svm.SVC(tol=1e-4, probability=True, kernel='linear', 
	decision_function_shape='ovr' ).fit(Xmat, Y)

pipe = make_pipeline(tfidf, clf)



te = TextExplainer(random_state=42)
te.fit(Xeval, pipe.predict_proba)
out = te.show_prediction(target_names=[0,1], feature_names=tfidf.get_feature_names())
#out = te.show_weights(target_names=[0,1])

print(out.data)
'''
'''

X_pmat = tfidf.transform(X_p).toarray()
print(X_pmat.shape, len(Y_p))

#https://medium.com/towards-artificial-intelligence/how-to-use-scikit-learn-eli5-library-to-compute-permutation-importance-9af131ece387
perm = PermutationImportance(clf).fit(X_pmat, Y_p)
out = eli5.show_weights(perm, feature_names=feature_names_alpha)


print(out.data)
'''


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