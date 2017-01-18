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
        self.field = DyadicPWConstant(c * np.ones([2,2]), 1)
    
class DyadicRandomField(object):
    """ Creates a random field on a dyadic subdivision of the unit cube """

    def __init__(self, div, a_bar, c, seed=None):
        # div is the subdivision power, not number

        self.div = div
        self.n_side = 2**div

        np.random.seed(seed)
        self.field = DyadicPWConstant(a_bar + c * (2.0 * np.random.random([self.n_side, self.n_side]) - 1.0), self.div)
        
    def make_fem_field(self, fem_div):
        # The field is defined to be constant on the dyadic squares
        # so here we generate a matrix of the values on all the dyadic squares at the
        # FEM mesh level, so that we can apply it in to the system easily
        
        fem_field = self.field.repeat(2**(fem_div - self.div), axis=0).repeat(2**(fem_div - self.div), axis=1)

        return fem_field

    def make_fem_field_flat(self, fem_div):
        # Just flatten!
        fem_field_flat = self.make_fem_field(fem_div).flatten()

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
        a = rand_field.field.interpolate(self.div)
        
        # Now we make the various diagonals
        diag = 2.0 * (a[:-1, :-1] + a[:-1,1:] + a[1:,:-1] + a[1:, 1:]).flatten()
        
        # min_diag is below the diagonal, hence deals with element to the left in the FEM grid
        lr_diag = -(a[1:, 1:] + a[:-1, 1:]).flatten()
        lr_diag[self.n_side-1::self.n_side] = 0 # These corresponds to edges on left or right extreme
        lr_diag = lr_diag[:-1]
        
        # Far min deals with the element that is above
        ud_diag = -(a[1:-1, 1:] + a[1:-1, :-1]).flatten()

        self.A = sparse.diags([diag, lr_diag, lr_diag, ud_diag, ud_diag], [0, -1, 1, -self.n_side, self.n_side]).tocsr()
        self.f = 0.5 * self.h * self.h * np.ones(self.n_el)

        self.u = DyadicPWLinear(np.zeros([self.n_side + 2, self.n_side + 2]), self.div)

    def solve(self):
        """ The bilinear form simply becomes \int_D a nab u . nab v = \int_D f """
        u_flat = sparse.linalg.spsolve(self.A, self.f)

        u = u_flat.reshape([self.n_side, self.n_side])
        # Pad the zeros on each side... (due to the boundary conditions) and make the 
        # dyadic piecewise linear function object
        self.u.values = np.pad(u, ((1,1),(1,1)), 'constant')


class DyadicPWConstant(object):
    """ Describes a piecewise linear function on a dyadic P1 tringulation of the unit cube.
        Includes routines to calculate L2 and H1 dot products, and interpolate between different dyadic levels
        """
    
    def __init__(self, values, div):

        self.div = div
        
        if (values.shape[0] != values.shape[1] and values.shape[0] != 2**div):
            raise Exception("Error - values must be on a dyadic square of size {0}".format(2**div))

        self.values = values
        
        # This grid is in the centers of the dyadic squares
        self.x_grid = np.linspace(0.0, 1.0, 2**self.div, endpoint=False) + 2.0**(-self.div - 1)
        self.y_grid = np.linspace(0.0, 1.0, 2**self.div, endpoint=False) + 2.0**(-self.div - 1)

    def interpolate(self, div):
        # The field is defined to be constant on the dyadic squares
        # so here we generate a matrix of the values on all the dyadic squares at the
        # finer mesh level, so that we can apply it in to the system easily
        
        if div >= self.div:
            return self.values.repeat(2**(div - self.div), axis=0).repeat(2**(div - self.div), axis=1)
        else:
            raise Exception('DyadicPWConstant: Interpolate div must be greater than or equal to field div')

    #TODO: Implement L2 and H1 with itself *AND* with the PW linear functions...

    
    def plot(self, ax, title='Dyadic piecewise constant function', alpha=0.5, cmap=cm.jet):

        # We do some tricks here (i.e. using np.repeat) to plot the piecewise constant nature of the random field...
        x = np.linspace(0.0, 1.0, 2**self.div + 1, endpoint = True).repeat(2)[1:-1]
        xs, ys = np.meshgrid(x, x)
        wframe = ax.plot_surface(xs, ys, self.values.repeat(2, axis=0).repeat(2, axis=1), cstride=1, rstride=1,
                                 cmap=cmap, alpha=alpha)
        ax.set_title(title)
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_axis_bgcolor('white')

class DyadicPWLinear(object):
    """ Describes a piecewise linear function on a dyadic P1 tringulation of the unit cube.
        Includes routines to calculate L2 and H1 dot products, and interpolate between different dyadic levels
        """

    def __init__(self, values, div):

        self.div = div
        
        if (values.shape[0] != values.shape[1] and values.shape[0] != 2**div + 1):
            raise Exception("Error - values must be on a dyadic square of size {0}".format(2**div))

        self.values = values
        # TODO: include a notion of the spatial grid, so that our interpolation
        # can be more general and we are not just stuck to the unit cube
        self.x_grid = np.linspace(0.0, 1.0, 2**self.div + 1, endpoint=True)
        self.y_grid = np.linspace(0.0, 1.0, 2**self.div + 1, endpoint=True)
    
    def H1_dot(self, f):
        """ Compute the H1_0 dot product with another DyadicPWLinear function
            automatically interpolates the coarser function """

        u, v, match_div = self.match_grids(f)

        h = 2.0**(-match_div)
        n_side = 2**match_div

        p = 2 * np.ones([n_side, n_side+1])
        p[:,0] = p[:,-1] = 1
        dot = (p * (u[1:,:] - u[:-1,:]) * (v[1:,:] - v[:-1,:])).sum()
        p = 2 * np.ones([n_side+1, n_side])
        p[0,:] = p[-1,:] = 1
        dot = dot + (p * (u[:,:-1] - u[:,1:]) * (v[:,:-1] - v[:,1:])).sum()

        return 0.5 * dot

    def L2_dot(self, f):
        """ Compute the L2 dot product with another DyadicPWLinear function,
            automatically interpolates the coarser function """
        
        u, v, match_div = self.match_grids(f)

        h = 2.0**(-match_div)
        n_side = 2**match_div

        # u and v are on the same grid / triangulation, so now we do the simple L2
        # inner product (hah... simple??)

        # the point adjacency matrix
        p = 6 * np.ones([n_side+1, n_side+1])
        p[:,0] = p[0,:] = p[:,-1] = p[-1,:] = 3 
        p[0,0] = p[-1,-1] = 1
        p[0,-1] = p[-1, 0] = 2 
        dot = (u * v * p).sum()
        
        # Now add all the vertical edges
        p = 2 * np.ones([n_side, n_side+1])
        p[0,:] = p[-1,:] = 1
        dot = dot + ((u[1:,:] * v[:-1,:] + u[:-1,:] * v[1:,:]) * p * 0.5).sum()

        # Now add all the horizontal edges
        p = 2 * np.ones([n_side+1, n_side])
        p[:,0] = p[:,-1] = 1
        dot = dot + ((u[:,1:] * v[:,:-1] + u[:,:-1] * v[:,1:]) * p * 0.5).sum()

        # Finally all the diagonals (note every diagonal is adjacent to two triangles,
        # so don't need p)
        dot = dot + (u[:-1,1:] * v[1:,:-1] + u[1:,:-1] * v[:-1,1:] ).sum()
        
        """ An element wise test of my matrix based calculation above...

        dot2 = 0.0
        for i in range(n_side):
            for j in range(n_side):

                dot2 = dot2 + ( u[i,j] * v[i,j] + u[i+1,j]*v[i+1,j] + u[i,j+1]*v[i,j+1]
                                            + 0.5 * (u[i,j]*v[i+1,j] + u[i+1,j] * v[i,j]
                                                    +u[i,j]*v[i,j+1] + u[i,j+1] * v[i,j]
                                                    +u[i,j+1]*v[i+1,j] + u[i+1,j]*v[i,j+1])
                                            + u[i+1,j]*v[i+1,j] + u[i,j+1]*v[i,j+1] + u[i+1,j+1]*v[i+1,j+1]
                                            + 0.5 * (u[i+1,j+1]*v[i+1,j] + u[i+1,j]*v[i+1,j+1]
                                                    +u[i+1,j+1]*v[i,j+1] + u[i,j+1]*v[i+1,j+1]
                                                    +u[i+1,j]*v[i,j+1] + u[i,j+1]*v[i+1,j]) ) """

        return h * h * dot / 12

    def match_grids(self, f):
        """ Check the dyadic division of f and adjust the coarser one,
            which we do through linear interpolation, returns two functions
            matched, with necessary interpolation, and the division
            level to which we interpolated """

        if self.div == f.div:
            return self.values, f.values, self.div
        if self.div > f.div:
            return self.values, f.interpolate(self.div), self.div
        if self.div < f.div:
            return self.interpolate(f.div), f.values, f.div

    def interpolate(self, interp_div):
        """ Simple interpolation routine to make this function on a finer division dyadic grid """
        
        if (interp_div < self.div):
            raise Exception("Interpolation division smaller than field division! Need to integrate")

        interp_func = scipy.interpolate.interp2d(self.x_grid, self.y_grid, self.values, kind='linear')

        x = y = np.linspace(0.0, 1.0, 2**interp_div + 1, endpoint=True)
        
        return interp_func(x, y)

    def plot(self, ax, title='Piecewise linear function', div_frame=4, alpha=0.5, cmap=cm.jet):

        x = np.linspace(0.0, 1.0, self.values.shape[0], endpoint = True)
        y = np.linspace(0.0, 1.0, self.values.shape[1], endpoint = True)
        xs, ys = np.meshgrid(x, y)

        if self.div > div_frame:
            wframe = ax.plot_surface(xs, ys, self.values, cstride=2**(self.div - div_frame), rstride=2**(self.div-div_frame), 
                                     cmap=cmap, alpha=alpha)
        else:
            wframe = ax.plot_surface(xs, ys, self.values, cstride=1, rstride=1, cmap=cmap, alpha=alpha)

        ax.set_axis_bgcolor('white')
        ax.set_xlabel('$x$')
        ax.set_ylabel('$y$')
        ax.set_title(title)

class Measurements(object):
    """ A measurement of the solution u of the PDE / FEM solution, in some linear subspace W """

class RandomPointMeasurements(Measurements):

    def __init__(self, i, j):
        self.i = i
        self.j = j

