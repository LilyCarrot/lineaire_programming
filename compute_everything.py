import time
import cplex
import load_data
import hinge
import hardmargin
import ramp

if __name__=='__main__':
  for filename in ['test_2.txt', 'test_3.txt', 'test_bc_orig.txt', 'bc-orig.txt',
                   'diabetes_small.txt', 'diabetes_100.txt', 'diabetes_140.txt', 'diabetes_157.txt',
                   'diabetes_160.txt','heart-bin.txt', 'diabetes_300.txt', 'diabetes.txt',
                   'liver.txt', 'sonar.txt']:
    (X, y) = load_data.load_data('data/' + filename)
    n = len(y)
    d = len(X[0])
    print('File: ', filename, '; n: ', n, '; d: ', d)
    for c in [0.001, 0.05, 0.5, 1.0, 4.0]:
      print('  c: ', c)
      p_h = hinge.createCplexInstance(d, n, X, y, c)
      p_hm = hardmargin.createCplexInstance(d, n, X, y, c)
      p_ramp = ramp.createCplexInstance(d, n, X, y, c)
      for (p, m) in [(p_h, 'hinge'), (p_hm, 'hm'), (p_ramp, 'ramp')]:
        p.write('lp/' + filename + '.' + m + str(c) + '.lp')
        try:
          t0 = time.time()
          p.solve()
          t1 = time.time()
          w = p.solution.get_values([j for j in range(d)])
          b = p.solution.get_values(d)
          print('    w: ', w, '; b: ', b, ';   (time: ', t1 - t0, ')')
          with open('wb/' + filename + '.' + m + str(c) + '.txt', 'w+') as f:
            print('    w: ', w, '; b: ', b, file=f)
            print('    time: ', t1 - t0, file=f)
          p.solution.write('sol/' + filename + '.' + m + str(c) + '.txt')
        except cplex.exceptions.errors.CplexSolverError:
          print('cplex.exceptions.errors.CplexSolverError')
      print()
