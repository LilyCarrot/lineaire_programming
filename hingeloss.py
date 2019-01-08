import numpy as np
import svm

# calculer les xi

# minimiser 1/2 ||w||^2 + C (sum xi)

def xi(w):
  x = [np.dot(w.T, l) for l in svm.A[:-2]]
  x = [(0 if c >= 1 else 1 - c) for c in x]
  return x

print(xi(svm.w0))

# COMPROMISE PARAMETER C
c = 5

def loss(w):
  return (svm.loss(w) + c*sum(xi(w)))

def jac(w):
  return (svm.jac(w) + c*sum(xi(w)))

leftMemberIneq = svm.minusOne

def xiplusplus(w):
  xpp = xi(w)
  xpp.append(0)
  xpp.append(0)
  return xpp

cons = {'type':'ineq',
        'fun':lambda w: svm.minusOne + xiplusplus(w) - np.dot(svm.A,w),
        'jac':lambda w: -svm.A + xiplusplus(w)}

svm.solve(loss, jac, cons)
