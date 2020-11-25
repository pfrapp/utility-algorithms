#!/usr/bin/env python3

#
# Author: Philipp Rapp
# Date: Nov-2020
#
# This is a small script containing some code to plot
# the ellipse corresponding to a covariance matrix.
#


import numpy as np
from numpy import deg2rad, rad2deg, sin, cos, arctan2, pi, sqrt, exp
from numpy.linalg import svd, inv, det
import matplotlib.pyplot as plt

def plot_covariance_matrix(Sigma):
    ''' Plot the one-sigma boundary of a given covariance matrix '''
    print('Plotting covariance matrix')
    print(Sigma)
    u, v, w = svd(Sigma)
    phi = arctan2(u[1,0], u[0,0])
    print('Reconstructed phi = {} deg'.format(rad2deg(phi)))
    print('Principal axes = {}'.format(v))
    # Compute points on the ellipse
    N = 100
    alpha = np.linspace(0, 2*pi, N)
    s, c = sin(phi), cos(phi)
    # Rotation matrix
    R = np.array([[c, -s], [s, c]])
    # Scaling matrix
    S = np.diag(sqrt(v))
    P = np.zeros((N, 2))
    for ii, a in enumerate(alpha):
        s, c = sin(a), cos(a)
        pos = np.array([c, s])
        # print('pos = ')
        # print(pos)
        pos = np.matmul(R, np.matmul(S, pos))
        P[ii,:] = pos
        
    
    plt.figure(1)
    plt.clf()
    plt.plot(P[:,0], P[:,1], label='One-sigma boundary')
    plt.grid(True)
    plt.legend()
    plt.title('Explicit computation')
    plt.show()
    
def plot_covariance_matrix_contour(Sigma):
    x = np.linspace(-5,5,200)
    y = np.linspace(-5,5,200)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(X.shape)
    G = np.zeros(X.shape)   # Gaussian
    for ii, (x_row, y_row) in enumerate(zip(X, Y)):
        for jj, (xx, yy) in enumerate(zip(x_row, y_row)):
            # print('Pos = ({}, {}) at indices ({}, {})'.format(xx, yy, ii, jj))
            v = np.array([xx, yy])
            Z[ii,jj] = np.dot(v, np.dot(inv(Sigma), v)) - 1
            exp_arg = -0.5 * (np.dot(v, np.dot(inv(Sigma), v)))
            G[ii,jj] = 1/sqrt(det(Sigma))**2 * exp(exp_arg)
            if xx <= -2.0 and yy <= -1.5:
                # Z[ii,jj] = -10
                pass
    
    plt.figure(2)
    plt.clf()
    plt.contourf(X, Y, Z)
    plt.grid(True)
    plt.colorbar()
    plt.title('Quadric countour')
    plt.show()
    
    plt.figure(3)
    plt.clf()
    plt.contourf(X, Y, G)
    plt.grid(True)
    plt.colorbar()
    plt.title('Gaussian countour')
    plt.show()
    


def main():
    Sigma = np.array([[5.0, 0.0], [0.0, 1.0]])
    # Construct a rotation matrix
    phi = deg2rad(10)
    s, c = sin(phi), cos(phi)
    R = np.array([[c, -s], [s, c]])
    Sigma = np.matmul(np.matmul(R, Sigma), R.transpose())
    # Sigma = np.array([[3.0,1.0], [1.0, 2.0]])
    plot_covariance_matrix((Sigma))
    plot_covariance_matrix_contour(Sigma)

                           
if __name__ == '__main__':
    main()
