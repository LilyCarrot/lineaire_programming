from sklearn.svm import LinearSVC

filename = 'data/test_bc_orig.txt'

def load_data(filename):
	X = [[]]
	y = []
	return (X, y)

(X, y) = load_data(filename)

clf = LinearSVC(random_state=0, tol=1e-5)
clf.fit(X, y)

print(clf.coef_)

print(clf.predict(X[0]))

