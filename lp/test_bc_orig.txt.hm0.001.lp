\ENCODING=ISO-8859-1
\Problem name: SVM

Minimize
 obj: 0.001 z0 + 0.001 z1 + [ w0 ^2 + w1 ^2 + w2 ^2 + w3 ^2 + w4 ^2 + w5 ^2
      + w6 ^2 + w7 ^2 + w8 ^2 ] / 2
Subject To
 c1: - 1.17503021 w0 - 0.700888296 w1 - 0.736211012 w2 - 0.626722131 w3
     - 0.531348307 w4 - 0.70282871 w5 - 0.569078192 w6 - 0.610515332 w7
     - 0.357098166 w8 + b + 10000 z0 >= 1
 c2: - 1.300518873 w0 - 2.341581549 w1 - 2.380246361 w2 - 2.73001473 w3
     - 1.240678809 w4 - 1.734438309 w5 - 2.638453438 w6 - 2.307976184 w7
     - 4.366756456 w8 - b + 10000 z1 >= 1
Bounds
-1 <= b <= 1
 0 <= z0 <= 1
 0 <= z1 <= 1
Binaries
 z0  z1 
End
