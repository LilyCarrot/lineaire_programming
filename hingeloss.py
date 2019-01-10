import numpy as np
from scipy import optimize
import svm


# minimiser 1/2 ||w||^2 + C (sum xi)

def newLineForHinge(oldline, i, n):
  zeroes = [0.0 for k in range(n)]
  if (i <= n-1):
    zeroes[i] = 1.0
  return (oldline + zeroes)

A = svm.A.tolist()
print(A)
A = [newLineForHinge(oldline, i, svm.n) for i,oldline in enumerate(A)]
print(A)
A = A + [[0 for k in range(svm.d+1)] + line for line in np.eye(svm.n).tolist()]
print(A)
print('2n+2 = ', end='')
print(2 * svm.n + 2)
print('hauteur(A) = ', end='')
print(len(A))
A = np.array(A, ndmin=2)
print(A)


# COMPROMISE PARAMETER C
c = 5

d = svm.d
def objective(V):
  return (np.dot(V[:d].T, V[:d]) + c * sum(V[d+1:]))

def jac(V):
  return (svm.jac(V[:d]) + c*sum(V[d+1:]))

minusOne = np.array([-1 for i in range(2 * svm.n + 2)])

cons = {'type':'ineq',
        'fun':lambda V: minusOne - np.dot(A,V),
        'jac':lambda V: -A}
opt = {'disp':False}

V0 = np.random.randn(d+1+svm.n)

def solve(objective, jac, cons):
  res = optimize.minimize(objective, V0, jac=jac, constraints=cons,
                          method='SLSQP', options=opt)
  #res_uncons = optimize.minimize(objective, V0, jac=jac, method='SLSQP',
  #                               options=opt)

if __name__ == '__main__':
  solve(objective, jac, cons)
