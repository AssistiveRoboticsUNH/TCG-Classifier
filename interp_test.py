from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import svm, metrics

import numpy as np

import eli5
from eli5.lime import TextExplainer
from eli5.sklearn import PermutationImportance

from sklearn.pipeline import Pipeline, make_pipeline

# Setup training/eval data

Xtrain = [
		"I am a cat",
		"What do i cat",
		"a cat is where its at",
		"how about that cat",
		"the cat there sat",

		"I am a dog",
		"What do i dog",
		"a dog is where its at",
		"how about that dog",
		"the dog there sat",

		"I am a rat",
		"What do i rat",
		"a rat is where its at",
		"how about that rat",
		"the rat there sat",
		]

'''
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
'''

#Ytrain = [0,0,0,0,0,1,1,1,1,1]
#Ytrain = [0,0,0,0,0,1,1,1,1,1, 1,1,1,1,1]#2,2,2,2,2]
Ytrain = [0,0,0,0,0,1,1,1,1,1, 2,2,2,2,2]

Xeval = [
		"it was a cat",
		"it was a dog",
		"it was a rat",

		"I watched cat where it sat",
		"I watched dog where it sat",
		"I watched rat where it sat",
				
			]



'''
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
'''

Yeval = [0,1,2,0,1,2]
#Yeval = [0,1,1,0,1,1,]#[0,1,2,0,1,2]#[0,0,0,0,1,1,1,1,2,2,2,2]
#Yeval = [0,1,0,1]#[0,0,0,0,1,1,1,1,2,2,2,2]


# define model
tfidf = CountVectorizer()
#tfidf = TfidfVectorizer(ngram_range=(1,1), sublinear_tf=True)
clf = svm.SVC(max_iter=100, tol=1e-4, probability=True, 
	kernel='linear', decision_function_shape='ovr' )

#clf = svm.LinearSVC( )

Xtrain_tfidf = tfidf.fit_transform(Xtrain)
clf.fit(Xtrain_tfidf, Ytrain)

# fit model
pred = clf.predict(tfidf.transform(Xeval))
print(metrics.accuracy_score(pred, Yeval))

# understanding features
fn_src = np.array(tfidf.get_feature_names())
vocab = tfidf.vocabulary_
print(fn_src)

coef = clf.coef_.toarray()
print(coef.shape)

for label in range(coef.shape[0]):

	print('')
	print("Label: ", label)

	cf = coef[label].reshape(-1)

	order = cf.argsort()
	cf = cf[order][::-1]
	fn = fn_src[:][order][::-1]

	for f, c in zip(fn, cf):
		print(f, vocab[f], c)

print(clf.intercept_)
print(tfidf.transform(Xeval).shape)
print(coef.shape)



print('--------------')
print('')

a = tfidf.transform(Xeval).toarray()
b = clf.coef_.toarray().T

print(a.shape, b.shape)

dec = np.dot(a, b)

print(dec.shape)
print(dec)
print(np.argmax(dec, axis = 1))

print(clf.decision_function(a))



print('--')

pred = dec < 0
conf = -dec
n_class = len(clf.classes_)
n_samples = dec.shape[0]

votes = np.zeros((n_samples, n_class))
sum_of_conf = np.zeros((n_samples, n_class))

k = 0
for i in range(n_class):
	for j in range(i+1, n_class):
		sum_of_conf[:, i] -= conf[:, k]
		sum_of_conf[:, j] += conf[:, k]
		votes[pred[:, k] == 0, i] += 1
		votes[pred[:, k] == 1, j] += 1
		k+=1

trans_conf = sum_of_conf / (3 * np.abs(sum_of_conf) +1)
out = votes + trans_conf

print(out)





print('--------------')
print('')

params = clf.get_params()
sv = clf.support_vectors_.toarray()
nv = clf.n_support_
a = clf.dual_coef_.toarray()
b = clf.intercept_
cs = fn_src
X  = tfidf.transform(Xeval).toarray()

print("sv:", sv.shape)
print(sv)

print("a:", a.shape)
print(a)

k = []
for vi in sv:
	print('')
	print(vi)
	print(np.dot(vi, X))
	k.append(np.dot(vi, X))
	


#k = [np.dot(vi, X) for vi in sv]

print("kernel:", len(k))#len(k), len(k[0]))
print(k)



#print("kernel:", k)
#print(nv)

# define the start and end index for support vectors for each class
start = [sum(nv[:i]) for i in range(len(nv))]
end = [start[i] + nv[i] for i in range(len(nv))]

print("start:", start)
print("end:", end)


# calculate: sum(a_p * k(x_p, x)) between every 2 classes



'''
print("a:", a.shape)
print(a)
#print("a[0]:", a[0])

for i in range(len(nv)):
	for j in range(i+1,len(nv)):
		for p in range(start[j], end[j]):
			print('')
			print(i, j, p)

			print(a[ i ][p])
			print(k[p])
			print(a[j-1][p])
'''
c = [ sum(a[ i ][p] * k[p] for p in range(start[j], end[j])) +
      sum(a[j-1][p] * k[p] for p in range(start[i], end[i]))
            for i in range(len(nv)) for j in range(i+1,len(nv))]

print("coeficients")
print(np.array(c))
print(clf.coef_.toarray())

# add the intercept
df = [sum(x) for x in zip(c, b)] 
print(np.array(df))

print(clf.decision_function(X))


'''
perm = PermutationImportance(clf).fit(tfidf.transform(Xeval).toarray(), Yeval)
out = eli5.show_weights(perm, feature_names=fn)

print(out.data)
'''

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