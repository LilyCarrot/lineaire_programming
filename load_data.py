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
    return (X, y)
