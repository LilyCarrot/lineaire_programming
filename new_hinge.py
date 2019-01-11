import sys
import numpy as np
from sklearn.svm import LinearSVC

def process_one_line(line):
  numbers = [float(word) for word in line.strip().split(',')]
  y = numbers[-1]
  x = numbers[:-1]
  d = len(x)
  return (d, x, y)

def load_data(filename):
  with open(filename, 'r') as f:
    X = []
    y = []
    for line in f:
      if len(line) > 3 and '@' not in line:
        (d, xi, yi) = process_one_line(line)
        X.append(xi)
        y.append(yi)
    X = np.array(X)
    y = np.array(y)
    return (X, y)

def main(lossname):
  if len(sys.argv) > 1:
    C = float(sys.argv[1])
  else:
    C = 1.0
  if len(sys.argv) > 2:
    filename = sys.argv[2]
  else:
    #filename = 'data/test_bc_orig.txt'
    filename = 'data/test_3.txt'
  (X, y) = load_data(filename)
  print("X:")
  print(X)
  print("y:")
  print(y)
  print()

  clf = LinearSVC(random_state=0, tol=1e-5, loss=lossname, C=2)
  clf.fit(X, y)

  print("w,b:")
  print(clf.coef_)
  print()

  print("classification:")
  nberr = 0
  for xi,yi in zip(X,y):
    print(xi, yi)
    zi = clf.predict([xi])
    print(zi)
    if yi != zi:
      nberr = nberr + 1
  print()
  print("err = ", nberr, '/', len(X))
  print()
  

if __name__ == '__main__':
  main('hinge')
