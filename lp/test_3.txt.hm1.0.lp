\ENCODING=ISO-8859-1
\Problem name: SVM

Minimize
 obj: z0 + z1 + z2 + z3 + z4 + z5 + z6 + z7 + z8 + [ w0 ^2 + w1 ^2 ] / 2
Subject To
 c1: w0 + w1 + b + 10000 z0 >= 1
 c2: 2 w0 + w1 + b + 10000 z1 >= 1
 c3: w0 + 3 w1 + b + 10000 z2 >= 1
 c4: - 2 w0 - 3 w1 - b + 10000 z3 >= 1
 c5: - 3 w0 - 2 w1 - b + 10000 z4 >= 1
 c6: w0 + 2 w1 + b + 10000 z5 >= 1
 c7: - 2 w0 - 2 w1 - b + 10000 z6 >= 1
 c8: - 3 w0 - 3 w1 - b + 10000 z7 >= 1
 c9: - 3 w0 - w1 - b + 10000 z8 >= 1
Bounds
-1 <= b <= 1
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
