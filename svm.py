import cplex

def process_one_line(line):
  numbers = [float(word) for word in line.strip().split(',')]
  y = 1.0 if numbers[-1] == 1 else -1.0
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
#    X = np.array(X)
#    y = np.array(y)
    return (X, y)

def createCplexInstance(d,n,X,y):
  p = cplex.Cplex()
  p.set_problem_name("SVM")
  p.objective.set_sense(p.objective.sense.minimize)
  my_colnames = [["w" + str(i) for i in range(d)], ["b"],
                        ["z" + str(i) for i in range(n)]]
  p.variables.add(types = [p.variables.type.continuous] * len(my_colnames[0]),
                  names = my_colnames[0])
  p.objective.set_quadratic([[[i],[1]] for i in range(d)])
  p.variables.add(obj=[0], types = p.variables.type.continuous, names="b")
  wxiplusb_left = createWxiplusbConstraints(d,n,X,y)
  p.linear_constraints.add(lin_expr=wxiplusb_left,
                           senses=['G' for i in range(d)],
                           rhs=wxiplusb_right)
  return p

