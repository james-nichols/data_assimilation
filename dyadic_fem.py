import math
import numpy as np
import scipy as sp
import scipy.signal
import scipy.special
import scipy.optimize
import scipy.interpolate
from scipy import sparse
from itertools import *

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm 
from mpl_toolkits.mplot3d import axes3d, Axes3D

import pdb

class ConstantField(object):

    def __init__(self, c=1.0):
        self.c = c
    
    def make_fem_field(self, fem_div):
        return self.c * np.ones([2**fem_div, 2**fem_div])


class DyadicField(object):
    """ Creates a random field on a dyadic subdivision of the unit cube """

    def __init__(self, div, a_bar, c, seed=None):
        # div is the subdivision power, not number

        self.div = div
        self.n_side = 2**div

        np.random.seed(seed)
        self.field = a_bar + c * (2.0 * np.random.random([self.n_side, self.n_side]) - 1.0)
        
    def make_fem_field(self, fem_div):
        # The field is defined to be constant on the dyadic squares
        # so here we generate a matrix of the values on all the dyadic squares at the
        # FEM mesh level, so that we can apply it in to the system easily
        
        fem_field = self.field.repeat(2**(fem_div - self.div), axis=0).repeat(2**(fem_div - self.div), axis=1)

        return fem_field

    def make_fem_field_flat(self, fem_div):
        # Just flatten!
        fem_field_flat = self.make_fem_field(fem_div).flatten()

    def plot(self):
        """ This routine just knows that it's piece-wise constant and can plot appropriately """
       
        
        """x = np.linspace(0.0, 1.0, self.n_side , endpoint = True) 
        #z_interp = scipy.interpolate.interp2d(x, x, self.u, kind='linear')
        xs, ys = np.meshgrid(x, x)

        fig = plt.figure()
        ax = Axes3D(fig)
        pdb.set_trace()
        div_frame = 5
        if self.div > div_frame:
            wframe = ax.plot_surface(xs, ys, self.field, cstride=2**(self.div - div_frame), rstride=2**(self.div-div_frame))
        else:
            wframe = ax.plot_surface(xs, ys, self.field)"""

        #plt.show() 
        #fig = plt.figure()
        sns.heatmap(self.field)


class DyadicPWLinear(object):
    """ Piecewise linear on the triangulated dyadic grid. This class automatically does
        integration on sup-grid areas, and interpolation """

class DyadicPWConstant(object):
    """ Piecewise constant on the dyadic grid. This class automatically does
        integration on sup-grid areas, and interpolation """

class DyadicFEMSolver(object):
    """ Solves the -div( a nabla u ) = f PDE on a grid, with a given by 
        some random / deterministic field, with dirichelet boundary conditions """

    def __init__(self, div, rand_field, f):
       
        self.div = div
        # -1 as we are interested in the grid points, of which there is an odd number
        self.n_side = 2**self.div - 1
        self.n_el = self.n_side * self.n_side
        self.h = 1.0 / (self.n_side + 1)

        # Makes an appropriate sized field for our FEM grid
        a = rand_field.make_fem_field(self.div)
        
        #A_test = np.zeros([self.n_side, self.n_side])
        #for i in range(self.n_side):
        #    A_test[i,i] = 
        #    for j in range(self.n_side):
        #        # Build the matrix by hand...
        #        A_test[i,j] =  

        # Now we make the various diagonals
        diag = 2.0 * (a[:-1, :-1] + a[:-1,1:] + a[1:,:-1] + a[1:, 1:]).flatten()
        
        # min_diag is below the diagonal, hence deals with element to the left in the FEM grid
        lr_diag = -(a[1:, 1:] + a[:-1, 1:]).flatten()
        lr_diag[self.n_side-1::self.n_side] = 0
        lr_diag = lr_diag[:-1]
        
        # Far min deals with the element that is above
        ud_diag = -(a[1:-1, 1:] + a[1:-1, :-1]).flatten()

        self.A = sparse.diags([diag, lr_diag, lr_diag, ud_diag, ud_diag], [0, -1, 1, -self.n_side, self.n_side]).tocsr()
        self.f = 0.5 * self.h * self.h * np.ones(self.n_el)

    def solve(self):
        """ The bilinear form simply becomes \int_D a nab u . nab v = \int_D f """
        u_flat = sparse.linalg.spsolve(self.A, self.f)

        self.u = u_flat.reshape([self.n_side, self.n_side])
        # Pad the zeros on each side... (due to the boundary conditions)
        self.u = np.pad(self.u, ((1,1),(1,1)), 'constant')

    def plot(self):

        x = np.linspace(0.0, 1.0, self.n_side + 2, endpoint = True) 
        #z_interp = scipy.interpolate.interp2d(x, x, self.u, kind='linear')
        xs, ys = np.meshgrid(x, x)

        fig = plt.figure()
        ax = Axes3D(fig)

        div_frame = 3
        if self.div > div_frame:
            wframe = ax.plot_surface(xs, ys, self.u, cstride=2**(self.div - div_frame), rstride=2**(self.div-div_frame), cmap=cm.jet)
        else:
            wframe = ax.plot_surface(xs, ys, self.u, cmap=cm.jet)

        #plt.show()

class Measurement(object):
    """ A measurement of the solution u of the PDE / FEM solution, in some linear subspace W """

class Approximation(object):
    """ A reconstruction """
