import numpy as np
from scipy import optimize

filename = 'data/bc-orig.txt'

def process_one_line(line):
  words = line.split(',')
  d = len(words) - 1
  x = [float(word) for word in words]
  y = x[-1]
  x = x[:-1]
  return (d, x, y)

def calculeLine(x, y):
  xlst = [- y * xi for xi in x]
  xlst.append(-y)
  return xlst

def addConstraintsForB(Alst, d):
  c1 = [0 for i in range(d)]
  c1[-1] = 1
  c2 = [0 for i in range(d)]
  c2[-1] = -1
  Alst.append(c1)
  Alst.append(c2)

def load_data(filename):
  with open(filename, 'r') as f:
    n = 0
    Alst = []
    for line in f:
      line = line.strip()
      old_d = None
      if len(line) > 3 and '@' not in line:
        d, x, y = process_one_line(line)
        if old_d is not None and d != old_d:
          print("\n\n\nLINE OF DIFFERENT SIZES ", old_d, " AND ", d, "\n\n\n")
          exit(-1)
        Alst.append(calculeLine(x, y))
        n = n + 1
        old_d = d
    addConstraintsForB(Alst, d)
    A = np.array([[cell for cell in line] for line in Alst], ndmin = 2)
    return (n, d, A)

(n, d, A) = load_data(filename)
minusOne = np.array([-1 for i in range(n+2)])

print(A[:10])

w0 = np.random.randn(d+1)

def loss(w):
  d = w.size
  return (np.dot(w.T, w) - w[d] * w[d])

def jac(w):
  d = w.size
  coeffs = [1 for i in range(d)]
  coeffs[-1] = 0
  return (np.dot(w.T, np.array(coeffs)))

cons = {'type':'ineq',
        'fun':lambda w: np.dot(A,w),
        'jac':lambda w: -A}

opt = {'disp':False}

def solve():

    res_cons = optimize.minimize(loss, w0, jac=jac,constraints=cons,
                                 method='SLSQP', options=opt)

    #res_uncons = optimize.minimize(loss, x0, jac=jac, method='SLSQP',
    #                               options=opt)

    print('\nConstrained:')
    print(res_cons)

    print('\nUnconstrained:')
    print(res_uncons)


solve()