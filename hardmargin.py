import numpy as np
import cplex
import load_data

# create all the  yi (w.xi + b) >= 1 - M zi constraints
def createWxiplusbConstraints(d, n, X, y):
  m = 10000.0
  left = []
  for i, xi in enumerate(X):
    wxi = [y[i] * xij for xij in xi]
    b = [y[i]]
    mzi = [0.0 for j in range(n)]
    mzi[i] = m
    lefti = [e for lst in [wxi, b, mzi] for e in lst]
    lefti = [[j for j in range(d + 1 + n)], lefti]
    left.append(lefti)
  right = [1.0 for i in range(n)]
  senses = ['G' for i in range(n)]
  return left, senses, right

# create the linear program
def createCplexInstance(d,n,X,y, c):
  p = cplex.Cplex()
  p.set_problem_name("SVM")
  p.objective.set_sense(p.objective.sense.minimize)
  my_colnames = [["w" + str(i) for i in range(d)],
                 ["b"],
                 ["z" + str(i) for i in range(n)]]
                        
  # variables w0,w2,...,w(d-1)
  p.variables.add(types = [p.variables.type.continuous] * len(my_colnames[0]),
                  names = my_colnames[0])
                  
  # 1/2 ||w||^2 first half of the objective function
  p.objective.set_quadratic([[[i],[1]] for i in range(d)])

  # variable b
  p.variables.add(obj=[0], types = p.variables.type.continuous, names="b", lb = [-1.0], ub = [1.0])

  # variables z0,z1,...,z(n-1) (binary)
  # c sum zi second half of the objective function
  p.variables.add(obj=[c] * len(my_colnames[2]),
                  types = [p.variables.type.binary] * len(my_colnames[2]),
                  names = my_colnames[2])
  
  # yi (w.xi + b) >= 1 - M zi constraints
  wxiplusb_left, wxiplusb_senses, wxiplusb_right = createWxiplusbConstraints(d, n, X, y)
  p.linear_constraints.add(lin_expr=wxiplusb_left, senses=wxiplusb_senses, rhs=wxiplusb_right)
  
  return p

