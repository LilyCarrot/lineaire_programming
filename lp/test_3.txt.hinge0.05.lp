\ENCODING=ISO-8859-1
\Problem name: SVM

Minimize
 obj: 0.05 ksi0 + 0.05 ksi1 + 0.05 ksi2 + 0.05 ksi3 + 0.05 ksi4 + 0.05 ksi5
      + 0.05 ksi6 + 0.05 ksi7 + 0.05 ksi8 + [ w0 ^2 + w1 ^2 ] / 2
Subject To
 c1: w0 + w1 + b + ksi0 >= 1
 c2: 2 w0 + w1 + b + ksi1 >= 1
 c3: w0 + 3 w1 + b + ksi2 >= 1
 c4: - 2 w0 - 3 w1 - b + ksi3 >= 1
 c5: - 3 w0 - 2 w1 - b + ksi4 >= 1
 c6: w0 + 2 w1 + b + ksi5 >= 1
 c7: - 2 w0 - 2 w1 - b + ksi6 >= 1
 c8: - 3 w0 - 3 w1 - b + ksi7 >= 1
 c9: - 3 w0 - w1 - b + ksi8 >= 1
Bounds
-1 <= b <= 1
End
