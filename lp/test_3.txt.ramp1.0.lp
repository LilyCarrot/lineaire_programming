\ENCODING=ISO-8859-1
\Problem name: SVM

Minimize
 obj: ksi0 + ksi1 + ksi2 + ksi3 + ksi4 + ksi5 + ksi6 + ksi7 + ksi8 + 2 z0
      + 2 z1 + 2 z2 + 2 z3 + 2 z4 + 2 z5 + 2 z6 + 2 z7 + 2 z8 + [ w0 ^2 + w1 ^2
      ] / 2
Subject To
 c1: w0 + w1 + b + ksi0 + 10000 z0 >= 1
 c2: 2 w0 + w1 + b + ksi1 + 10000 z1 >= 1
 c3: w0 + 3 w1 + b + ksi2 + 10000 z2 >= 1
 c4: - 2 w0 - 3 w1 - b + ksi3 + 10000 z3 >= 1
 c5: - 3 w0 - 2 w1 - b + ksi4 + 10000 z4 >= 1
 c6: w0 + 2 w1 + b + ksi5 + 10000 z5 >= 1
 c7: - 2 w0 - 2 w1 - b + ksi6 + 10000 z6 >= 1
 c8: - 3 w0 - 3 w1 - b + ksi7 + 10000 z7 >= 1
 c9: - 3 w0 - w1 - b + ksi8 + 10000 z8 >= 1
Bounds
-1 <= b <= 1
 0 <= ksi0 <= 2
 0 <= ksi1 <= 2
 0 <= ksi2 <= 2
 0 <= ksi3 <= 2
 0 <= ksi4 <= 2
 0 <= ksi5 <= 2
 0 <= ksi6 <= 2
 0 <= ksi7 <= 2
 0 <= ksi8 <= 2
 0 <= z0 <= 1
 0 <= z1 <= 1
 0 <= z2 <= 1
 0 <= z3 <= 1
 0 <= z4 <= 1
 0 <= z5 <= 1
 0 <= z6 <= 1
 0 <= z7 <= 1
 0 <= z8 <= 1
Binaries
 z0  z1  z2  z3  z4  z5  z6  z7  z8 
End
