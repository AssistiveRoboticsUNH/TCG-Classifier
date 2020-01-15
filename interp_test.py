from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm


corpus = [
		"The cat (Felis catus) is a small carnivorous mammal.",
		"It is the only domesticated species in the family Felidae and often referred to as the domestic cat to distinguish it from wild members of the family.",
		"The cat is either a house cat, a farm cat or a feral cat",
		"Domestic cats are valued by humans for companionship and for their ability to hunt rodents.", 
		"About cat breeds are recognized by various cat registries.",
		
		"The domestic dog (Canis lupus familiaris when considered a subspecies of the wolf",
		"or Canis familiaris when considered a distinct species) is a member of the genus Canis (canines)",
		"which forms part of the wolf-like canids, and is the most widely abundant terrestrial carnivore.",
		"The dog and the extant gray wolf are sister taxa as modern wolves are not closely related to the wolves",
		"that were first domesticated, which implies that the direct ancestor of the dog is extinct. "]
Y = [0,0,0,0,0,1,1,1,1,1]

tfidf = TfidfVectorizer(sublinear_tf=True)

X = tfidf.fit_transform(corpus)

print(X.shape)

clf = svm.SVC(max_iter=1000, tol=1e-4, probability=True, 
	kernel='linear', decision_function_shape='ovr' ).fit(X, Y)
		
importance = clf.coef_
feature_names = tfidf.get_feature_names()

for i, f in enumerate(feature_names):
	print(f, clf.coef_[0][i], clf.coef_[1][i])
