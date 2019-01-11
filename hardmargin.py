import numpy as np
import cplex
import load_data

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

def createCplexInstance(d,n,X,y, c):
  p = cplex.Cplex()
  p.set_problem_name("SVM")
  p.objective.set_sense(p.objective.sense.minimize)
  my_colnames = [["w" + str(i) for i in range(d)], ["b"],
                        ["z" + str(i) for i in range(n)]]
  p.variables.add(types = [p.variables.type.continuous] * len(my_colnames[0]),
                  names = my_colnames[0])
  p.objective.set_quadratic([[[i],[1]] for i in range(d)])
  p.variables.add(obj=[0], types = p.variables.type.continuous, names="b", lb = [-1.0], ub = [1.0])
  p.variables.add(obj=[c] * len(my_colnames[2]),
                  types = [p.variables.type.binary] * len(my_colnames[2]),
                  names = my_colnames[2])
  wxiplusb_left, wxiplusb_senses, wxiplusb_right = createWxiplusbConstraints(d, n, X, y)
  p.linear_constraints.add(lin_expr=wxiplusb_left, senses=wxiplusb_senses, rhs=wxiplusb_right)
  return p

def addHardmarginToCplexInstance(p, d,n,X,y,c):
    p.variables.add(obj=[c] * len(my_colnames[2]),
                  types = [p.variables.type.binary] * len(my_colnames[2]),
                  names = my_colnames[2])
    return p

if __name__=='__main__':
  filename = 'data/test_3.txt'
  (X, y) = load_data.load_data(filename)
  n = len(y)
  d = len(X[0])
  c = 4.0
  
  print("Creating Cplex instance...")
#  p = load_data.createCplexInstance(d, n, X, y)
#  p = addHardmarginToCplexInstance(p, d,n,X,y,c)
  p = createCplexInstance(d, n, X, y, c)
  print(p)
  p.write("cplex_hardmargin.lp")
  print()
  
  print("Solving...")
  p.solve()
  print()
  
  print("Solution:")
  print(p.solution)
  p.solution.write("solutions_hardmargin.lp")
  print()
