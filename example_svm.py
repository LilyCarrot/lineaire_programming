import numpy as np
from scipy import optimize

# minimize
#     F = x[1]^2 + 4x[2]^2 -32x[2] + 64

#     F = w[1]^2 + ... + w[9]^2

#     
#
#

# subject to:
#      x[1] + x[2] <= 7
#     -x[1] + 2x[2] <= 4
#      x[1] >= 0
#      x[2] >= 0
#      x[2] <= 4

# in matrix notation:
#     F = (1/2)*x.T*H*x + c*x + c0

# subject to:
#     Ax <= b

# where:
#     H = [[2, 0],
#          [0, 8]]

#     c = [0, -32]

#     c0 = 64

#     A = [[ 1, 1],
#          [-1, 2],
#          [-1, 0],
#          [0, -1],
#          [0,  1]]

#     b = [7,4,0,0,4]

H = np.array([[2., 0.],
              [0., 8.]])

c = np.array([0, -32])

c0 = 64

A = np.array([[ 1., 1.],
              [-1., 2.],
              [-1., 0.],
              [0., -1.],
              [0.,  1.]])

b = np.array([7., 4., 0., 0., 4.])

w0 = np.random.randn(10)

def loss(x, sign=1.):
    return sign * (0.5 * np.dot(x.T, np.dot(H, x))+ np.dot(c, x) + c0)

def jac(x, sign=1.):
    return sign * (np.dot(x.T, H) + c)

cons = {'type':'ineq',
        'fun':lambda x: b - np.dot(A,x),
        'jac':lambda x: -A}

opt = {'disp':False}

def solve():

    res_cons = optimize.minimize(loss, x0, jac=jac,constraints=cons,
                                 method='SLSQP', options=opt)

    #res_uncons = optimize.minimize(loss, x0, jac=jac, method='SLSQP',
    #                               options=opt)

    print('\nConstrained:')
    print(res_cons)

    print('\nUnconstrained:')
    print(res_uncons)


solve()