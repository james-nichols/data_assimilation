"""
dyadic_fem.py

Author: James Ashton Nichols
Start date: January 2017

A library of tools to do state estimation of simple parametric PDEs. Includes
FEM solver, optimal state reconstruction, favorable basis computation,
and more. Links to relevant academic papers in README.md
"""

import math
import numpy as np
import scipy as sp
import scipy.signal
import scipy.special
import scipy.optimize
import scipy.interpolate
import scipy.linalg
from scipy import sparse
from itertools import *
import inspect
import copy

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm 
from mpl_toolkits.mplot3d import axes3d, Axes3D

import point_generator as pg

import pdb

class ConstantField(object):

    def __init__(self, c=1.0):
        self.c = c
        self.field = DyadicPWConstant(c * np.ones([2,2]), 1)
    
def make_dyadic_random_field(div, a_bar, c, seed=None):
    # div is the subdivision power, not number

    n_side = 2**div

    if seed is not None:
        np.random.seed(seed)
    
    return DyadicPWConstant(a_bar + c * (2.0 * np.random.random([n_side, n_side]) - 1.0), div)


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
        a = rand_field.interpolate(self.div).values
        
        # Now we make the various diagonals
        diag = 2.0 * (a[:-1, :-1] + a[:-1,1:] + a[1:,:-1] + a[1:, 1:]).flatten()
        
        # min_diag is below the diagonal, hence deals with element to the left in the FEM grid
        lr_diag = -(a[1:, 1:] + a[:-1, 1:]).flatten()
        lr_diag[self.n_side-1::self.n_side] = 0 # These corresponds to edges on left or right extreme
        lr_diag = lr_diag[:-1]
        
        # Far min deals with the element that is above
        ud_diag = -(a[1:-1, 1:] + a[1:-1, :-1]).flatten()

        self.A = sparse.diags([diag, lr_diag, lr_diag, ud_diag, ud_diag], [0, -1, 1, -self.n_side, self.n_side]).tocsr()
        self.f = f * 0.5 * self.h * self.h * np.ones(self.n_el)

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
    
    def __init__(self, values = None, div = None, func = None):

        if div is not None:
            self.div = div
        
            # This grid is in the centers of the dyadic squares
            self.x_grid = np.linspace(0.0, 1.0, 2**self.div, endpoint=False) + 2.0**(-self.div-1)
            self.y_grid = np.linspace(0.0, 1.0, 2**self.div, endpoint=False) + 2.0**(-self.div-1)
            
            if func is not None:
                if values is not None:
                    raise Exception('DyadicPWConstant: Specify either a function or the values, not both')
                x, y = np.meshgrid(self.x_grid, self.y_grid)
                self.values = func(x, y)
            elif values is not None:
                if (values.shape[0] != values.shape[1] or values.shape[0] != 2**div):
                    raise Exception("DyadicPWConstant: Error - values must be on a dyadic square of size {0}".format(2**div))
                self.values = values
            else:
                self.values = np.zeros([2**self.div, 2**self.div])
        else:            
            if values is not None:
                self.values = values
                self.div = int(math.log(values.shape[0], 2))
                if (values.shape[0] != values.shape[1] or values.shape[0] != 2**self.div):
                    raise Exception("DyadicPWConstant: Error - values must be on a dyadic square, shape of {0} closest to div {1}".format(2**self.div, self.div))
                self.x_grid = np.linspace(0.0, 1.0, 2**self.div, endpoint=False) + 2.0**(-self.div-1)
                self.y_grid = np.linspace(0.0, 1.0, 2**self.div, endpoint=False) + 2.0**(-self.div-1)
            elif func is not None:
                raise Exception('DyadicPWLinear: Error - need grid size when specifying function')

    def dot(self, other, space='H1'):
        if isinstance(other, type(self)):
            if space == 'L2':
                return self.L2_dot(other)
            elif space == 'H1':
                return self.L2_dot(other)
            else:
                raise Exception('Unrecognised Hilbert space norm ' + space)
        elif isinstance(other, DyadicPWLinear):
            return other.dot(self)
        else:
            raise Exception('Dot product can only be between compatible dyadic functions')

    def norm(self, space='H1'):
        return self.dot(self, space)
    
    def L2_dot(self, other):

        d = max(self.div, other.div)
        u = self.interpolate(d)
        v = other.interpolate(d)

        return (u.values * v.values).sum() * 2**(-2 * d)

    def interpolate(self, div):
        # The field is defined to be constant on the dyadic squares
        # so here we generate a matrix of the values on all the dyadic squares at the
        # finer mesh level, so that we can apply it in to the system easily
        
        if div < self.div:
            raise Exception('DyadicPWConstant: Interpolate div must be greater than or equal to field div')
        elif div == self.div:
            return self
        else:
            return DyadicPWConstant(values=self.values.repeat(2**(div-self.div), axis=0).repeat(2**(div-self.div), axis=1),
                                    div=div)

    #TODO: Implement L2 and H1 with itself *AND* with the PW linear functions...
   
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

    def plot(self, ax, title=None, alpha=0.5, cmap=cm.jet, show_axes_labels=True):

        # We do some tricks here (i.e. using np.repeat) to plot the piecewise constant nature of the random field...
        x = np.linspace(0.0, 1.0, 2**self.div + 1, endpoint = True).repeat(2)[1:-1]
        xs, ys = np.meshgrid(x, x)
        wframe = ax.plot_surface(xs, ys, self.values.repeat(2, axis=0).repeat(2, axis=1), cstride=1, rstride=1,
                                 cmap=cmap, alpha=alpha)
        
        ax.set_axis_bgcolor('white')
        if show_axes_labels:
            ax.set_xlabel('$x$')
            ax.set_ylabel('$y$')
        if title is not None:
            ax.set_title(title)

    # Here we overload the + += - -= * and / operators
    
    def __add__(self,other):
        if isinstance(other, DyadicPWConstant):
            d = max(self.div,other.div)
            u = self.interpolate(d)
            v = other.interpolate(d)
            return DyadicPWConstant(u.values + v.values, d)
        if isinstance(other, DyadicPWLinear):
            d = max(self.div,other.div)
            u = self.interpolate(d)
            v = other.interpolate(d)
            return DyadicPWHybrid(l_values=v.values, c_values=u.values, div=d)
        else:
            return DyadicPWConstant(self.values + other, self.div)

    __radd__ = __add__

    def __iadd__(self,other):
        if isinstance(other, DyadicPWConstant):
            d = max(self.div,other.div)
            u = self.interpolate(d)
            v = other.interpolate(d)
            self.div = d
            self.values = u.values + v.values
        else:
            self.values = self.values + other
        return self
        
    def __sub__(self,other):
        if isinstance(other, DyadicPWConstant):
            d = max(self.div,other.div)
            u = self.interpolate(d)
            v = other.interpolate(d)
            return DyadicPWConstant(u.values - v.values, d)
        if isinstance(other, DyadicPWLinear):
            d = max(self.div,other.div)
            u = self.interpolate(d)
            v = other.interpolate(d)
            return DyadicPWHybrid(l_values=-v.values, c_values=u.values, div=d)        
        else:
            return DyadicPWConstant(self.values - other, self.div)
    __rsub__ = __sub__

    def __isub__(self,other):
        if isinstance(other, DyadicPWConstant):
            d = max(self.div,other.div)
            u = self.interpolate(d)
            v = other.interpolate(d)
            self.div = d
            self.values = u.values - v.values
        else:
            self.values = self.values - other
        return self

    def __mul__(self,other):
        if isinstance(other, DyadicPWConstant):
            d = max(self.div,other.div)
            u = self.interpolate(d)
            v = other.interpolate(d)
            return DyadicPWConstant(u.values * v.values, d)
        else:
            return DyadicPWConstant(self.values * other, self.div)
    __rmul__ = __mul__

    def __pow__(self,power):
        return DyadicPWConstant(self.values**power, self.div)

    def __truediv__(self,other):
        if isinstance(other, DyadicPWConstant):
            d = max(self.div,other.div)
            u = self.interpolate(d)
            v = other.interpolate(d)
            return DyadicPWConstant(u.values / v.values, d)
        else:
            return DyadicPWConstant(self.values / other, self.div)

class DyadicPWLinear(object):
    """ Describes a piecewise linear function on a dyadic P1 tringulation of the unit cube.
        Includes routines to calculate L2 and H1 dot products, and interpolate between different dyadic levels
        """

    def __init__(self, values = None, div = None, func = None):
        
        if div is not None:
            self.div = div
            self.x_grid = np.linspace(0.0, 1.0, 2**self.div + 1, endpoint=True)
            self.y_grid = np.linspace(0.0, 1.0, 2**self.div + 1, endpoint=True)
            
            if func is not None:
                if values is not None:
                    raise Exception('DyadicPWLinear: Specify either a function or the values, not both')
                x, y = np.meshgrid(self.x_grid, self.y_grid)
                self.values = func(x, y)
            elif values is not None:
                if (values.shape[0] != values.shape[1] or values.shape[0] != 2**div + 1):
                    raise Exception("DyadicPWLinear: Error - values must be on a dyadic square of size {0}".format(2**div+1))
                self.values = values

            else:
                self.values = np.zeros([2**self.div + 1, 2**self.div + 1])
        else:
            if values is not None:
                self.values = values
                self.div = int(math.log(values.shape[0] - 1, 2))
                if (values.shape[0] != values.shape[1] or values.shape[0] != 2**self.div + 1):
                    raise Exception("DyadicPWLinear: Error - values must be on a dyadic square, shape of {0} closest to div {1}".format(2**self.div, self.div))
                self.x_grid = np.linspace(0.0, 1.0, 2**self.div + 1, endpoint=True)
                self.y_grid = np.linspace(0.0, 1.0, 2**self.div + 1, endpoint=True)
            elif func is not None:
                raise Exception('DyadicPWLinear: Error - need grid size when specifying function')

            # else: nothing is set up

            # TODO: keep the function so we can do a proper interpolation, not just a linear
            # interpolation... but maybe we don't want that either

    def dot(self, other, space='H1'):
        if isinstance(other, type(self)):
            if space == 'L2':
                return self.L2_dot(other)
            elif space == 'H1':
                return self.H1_dot(other)
            else:
                raise Exception('Unrecognised Hilbert space norm ' + space)
        elif isinstance(other, DyadicPWConstant):
            if space == 'L2':
                return self.L2_pwconst_dot(other)
            elif space == 'H1':
                # H1 norm between pw constant function same as L2 as first deriv is 0
                # almost everywhere...
                return self.L2_pwconst_dot(other)
            else:
                raise Exception('Unrecognised Hilbert space norm ' + space)
        else:
            raise Exception('Dot product can only be between compatible dyadic functions')

    def norm(self, space='H1'):
        return self.dot(self, space)
    
    def H1_dot(self, other):
        """ Compute the H1_0 dot product with another DyadicPWLinear function
            automatically interpolates the coarser function """
        
        d = max(self.div,other.div)
        u = self.interpolate(d).values
        v = other.interpolate(d).values

        h = 2.0**(-d)
        n_side = 2**d

        # This is du/dy
        p = 2 * np.ones([n_side, n_side+1])
        p[:,0] = p[:,-1] = 1
        dot = (p * (u[:-1,:] - u[1:,:]) * (v[:-1,:] - v[1:,:])).sum()
        # And this is du/dx
        p = 2 * np.ones([n_side+1, n_side])
        p[0,:] = p[-1,:] = 1
        dot = dot + (p * (u[:,1:] - u[:,:-1]) * (v[:,1:] - v[:,:-1])).sum()
        
        return 0.5 * dot + self.L2_inner(u,v,h)

    def L2_dot(self, other):
        """ Compute the L2 dot product with another DyadicPWLinear function,
            automatically interpolates the coarser function """
        
        d = max(self.div,other.div)
        u = self.interpolate(d).values
        v = other.interpolate(d).values
        
        h = 2.0**(-d)

        return self.L2_inner(u,v,h)

    def L2_inner(self, u, v, h):
        # u and v are on the same grid / triangulation, so now we do the simple L2
        # inner product (hah... simple??)

        # the point adjacency matrix
        p = 6 * np.ones(u.shape)
        p[:,0] = p[0,:] = p[:,-1] = p[-1,:] = 3
        p[0,0] = p[-1,-1] = 1
        p[0,-1] = p[-1, 0] = 2 
        dot = (u * v * p).sum()
        
        # Now add all the vertical edges
        p = 2 * np.ones([u.shape[0]-1, u.shape[1]])
        p[0,:] = p[-1,:] = 1
        dot = dot + ((u[1:,:] * v[:-1,:] + u[:-1,:] * v[1:,:]) * p * 0.5).sum()

        # Now add all the horizontal edges
        p = 2 * np.ones([u.shape[0], u.shape[1]-1])
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

    def L2_pwconst_dot(self, other):
        """ Compute the L2 dot product with a DyadicPWConstant function,
            automatically interpolates the coarser function """
        
        # NB that the v field is the piecewise constant function
        d = max(self.div, other.div)
        u = self.interpolate(d).values
        v = other.interpolate(d).values
        
        h = 2.0**(-d)
    
        # Top left triangle
        dot = ((u[:-1,:-1] + u[1:,:-1] + u[:-1,1:]) * v).sum()
        # Bottom left triangle
        dot += ((u[1:,1:] + u[1:,:-1] + u[:-1,1:]) * v).sum()

        return h * h * dot / 6


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
        
        if interp_div < self.div:
            raise Exception("Interpolation division smaller than field division! Need to integrate")
        elif interp_div == self.div:
            return self
        else:
            interp_func = scipy.interpolate.interp2d(self.x_grid, self.y_grid, self.values, kind='linear')
            x = y = np.linspace(0.0, 1.0, 2**interp_div + 1, endpoint=True)
            return DyadicPWLinear(interp_func(x, y), interp_div)

    def plot(self, ax, title=None, div_frame=4, alpha=0.5, cmap=cm.jet, show_axes_labels=True):

        x = np.linspace(0.0, 1.0, self.values.shape[0], endpoint = True)
        y = np.linspace(0.0, 1.0, self.values.shape[1], endpoint = True)
        xs, ys = np.meshgrid(x, y)

        if self.div > div_frame:
            wframe = ax.plot_surface(xs, ys, self.values, cstride=2**(self.div - div_frame), rstride=2**(self.div-div_frame), 
                                     cmap=cmap, alpha=alpha)
        else:
            wframe = ax.plot_surface(xs, ys, self.values, cstride=1, rstride=1, cmap=cmap, alpha=alpha)

        ax.set_axis_bgcolor('white')
        if show_axes_labels:
            ax.set_xlabel('$x$')
            ax.set_ylabel('$y$')
        if title is not None:
            ax.set_title(title)

    # Here we overload the + += - -= * and / operators
    
    def __add__(self,other):
        if isinstance(other, DyadicPWLinear):
            d = max(self.div,other.div)
            u = self.interpolate(d)
            v = other.interpolate(d)
            return DyadicPWLinear(u.values + v.values, d)
        if isinstance(other, DyadicPWConstant):
            d = max(self.div,other.div)
            u = self.interpolate(d)
            v = other.interpolate(d)
            return DyadicPWHybrid(l_values=u.values, c_values=v.values, div=d)
        else:
            return DyadicPWLinear(self.values + other, self.div)

    __radd__ = __add__

    def __iadd__(self,other):
        if isinstance(other, DyadicPWLinear):
            d = max(self.div,other.div)
            u = self.interpolate(d)
            v = other.interpolate(d)
            self.div = d
            self.values = u.values + v.values
        else:
            self.values = self.values + other
        return self
        
    def __sub__(self,other):
        if isinstance(other, DyadicPWLinear):
            d = max(self.div,other.div)
            u = self.interpolate(d)
            v = other.interpolate(d)
            return DyadicPWLinear(u.values - v.values, d)
        if isinstance(other, DyadicPWConstant):
            d = max(self.div,other.div)
            u = self.interpolate(d)
            v = other.interpolate(d)
            return DyadicPWHybrid(l_values=u.values, c_values=-v.values, div=d)
        else:
            return DyadicPWLinear(self.values - other, self.div)
    __rsub__ = __sub__

    def __isub__(self,other):
        if isinstance(other, DyadicPWLinear):
            d = max(self.div,other.div)
            u = self.interpolate(d)
            v = other.interpolate(d)
            self.div = d
            self.values = u.values - v.values
        else:
            self.values = self.values - other
        return self

    def __mul__(self,other):
        if isinstance(other, DyadicPWLinear):
            d = max(self.div,other.div)
            u = self.interpolate(d)
            v = other.interpolate(d)
            return DyadicPWLinear(u.values * v.values, d)
        else:
            return DyadicPWLinear(self.values * other, self.div)
    __rmul__ = __mul__

    def __pow__(self,power):
        return DyadicPWLinear(self.values**power, self.div)

    def __truediv__(self,other):
        if isinstance(other, DyadicPWLinear):
            d = max(self.div,other.div)
            u = self.interpolate(d)
            v = other.interpolate(d)
            return DyadicPWLinear(u.values / v.values, d)
        else:
            return DyadicPWLinear(self.values / other, self.div)


class DyadicPWHybrid(object):
    """ Describes a piecewise linear function on a dyadic P1 tringulation of the unit cube.
        Includes routines to calculate L2 and H1 dot products, and interpolate between different dyadic levels
        """

    def __init__(self, l_values=None, c_values=None, div=None):
        
        if div is not None:
            self.div = div
            self.x_grid = np.linspace(0.0, 1.0, 2**self.div + 1, endpoint=True)
            self.y_grid = np.linspace(0.0, 1.0, 2**self.div + 1, endpoint=True)
            
            if l_values is not None:
                if (l_values.shape[0] != l_values.shape[1] or l_values.shape[0] != 2**div + 1):
                    raise Exception("DyadicPWHybrid: Error - values must be on a dyadic square of size {0}".format(2**div+1))
                self.l_values = l_values
            else:
                self.l_values = np.zeros([2**self.div + 1, 2**self.div + 1])
            if c_values is not None:
                if (c_values.shape[0] != c_values.shape[1] or c_values.shape[0] != 2**div):
                    raise Exception("DyadicPWHybrid: Error - values must be on a dyadic square of size {0}".format(2**div+1))
                self.c_values = c_values
            else:
                self.c_values = np.zeros([2**self.div, 2**self.div])


        else:
            if l_values is not None:
                self.l_values = l_values
                self.div = int(math.log(l_values.shape[0] - 1, 2))
                if (l_values.shape[0] != l_values.shape[1] or l_values.shape[0] != 2**self.div + 1):
                    raise Exception("DyadicPWHybrid: Error - linear values must be on a dyadic square, shape of {0} closest to div {1}".format(2**self.div+1, self.div))
                self.x_grid = np.linspace(0.0, 1.0, 2**self.div + 1, endpoint=True)
                self.y_grid = np.linspace(0.0, 1.0, 2**self.div + 1, endpoint=True)
            elif c_values is not None:
                self.c_values = c_values
                self.div = int(math.log(c_values.shape[0], 2))
                if (c_values.shape[0] != c_values.shape[1] or c_values.shape[0] != 2**self.div):
                    raise Exception("DyadicPWHybrid: Error - constant values must be on a dyadic square, shape of {0} closest to div {1}".format(2**self.div, self.div))
            else:
                raise Exception("DyadicPWHybrid: Need either div or field values to inisialise!")

    def match_grids(self, f):
        """ Check the dyadic division of f and adjust the coarser one,
            which we do through linear interpolation, returns two functions
            matched, with necessary interpolation, and the division
            level to which we interpolated """

        if self.div == f.div:
            return self, f, self.div
        if self.div > f.div:
            return self, f.interpolate(self.div), self.div
        if self.div < f.div:
            return self.interpolate(f.div), f, f.div

    def norm(self, space='H1'):
        return self.dot(self, space)

    def dot(self, other, space='H1'):
        return 0.0
    

    def L2_dot(self, other):
        """ Compute the L2 dot product with another DyadicPWLinear function,
            automatically interpolates the coarser function """
        
        d = max(self.div,other.div)
        u = self.interpolate(d)
        v = other.interpolate(d)

        return self.L2_inner(u.l_values, v.l_values, u.c_values, v.c_values, d)

    def L2_inner(self, u, v, p, q, d):
        # u and v are linear values on the same grid / triangulation, 
        # while p and q are the constant values

        h = 2.0**(-d)

        # We must repeat all internal points twice in the linear fields
        t_u = u.repeat(2, axis=0).repeat(2, axis=1)[1:-1,1:-1] + p.repeat(2, axis=0).repeat(2, axis=1)
        t_v = v.repeat(2, axis=0).repeat(2, axis=1)[1:-1,1:-1] + q.repeat(2, axis=0).repeat(2, axis=1)

        # Now we can do a calculation a bit like the one for the PW linear function, but with a stride
        # which we include in the point adjacency matrix
        p = np.tile([[1,2],[2,1]], (2**d, 2**d))

        # the point adjacency matrix
        p = 6 * np.ones(u.shape)
        p[:,0] = p[0,:] = p[:,-1] = p[-1,:] = 3
        p[0,0] = p[-1,-1] = 1
        p[0,-1] = p[-1, 0] = 2 
        dot = (u * v * p).sum()
        
        # Now add all the vertical edges
        p = 2 * np.ones([u.shape[0]-1, u.shape[1]])
        p[0,:] = p[-1,:] = 1
        dot = dot + ((u[1:,:] * v[:-1,:] + u[:-1,:] * v[1:,:]) * p * 0.5).sum()

        # Now add all the horizontal edges
        p = 2 * np.ones([u.shape[0], u.shape[1]-1])
        p[:,0] = p[:,-1] = 1
        dot = dot + ((u[:,1:] * v[:,:-1] + u[:,:-1] * v[:,1:]) * p * 0.5).sum()

        # Finally all the diagonals (note every diagonal is adjacent to two triangles,
        # so don't need p)
        dot = dot + (u[:-1,1:] * v[1:,:-1] + u[1:,:-1] * v[:-1,1:] ).sum()

        return h * h * dot / 12

    def interpolate(self, interp_div):
        """ Simple interpolation routine to make this function on a finer division dyadic grid """
        
        if interp_div < self.div:
            raise Exception("Interpolation division smaller than field division! Need to integrate")
        elif interp_div == self.div:
            return self
        else:
            const = self.c_values.repeat(2**(interp_div-self.div), axis=0).repeat(2**(interp_div-self.div), axis=1)
            interp_func = scipy.interpolate.interp2d(self.x_grid, self.y_grid, self.l_values, kind='linear')
            x = y = np.linspace(0.0, 1.0, 2**interp_div + 1, endpoint=True)
            return DyadicPWHybrid(l_values = interp_func(x, y), c_values = const, div = interp_div)

    def plot(self, ax, title=None, div_frame=4, alpha=0.5, cmap=cm.jet, show_axes_labels=True):

        # We do some tricks here (i.e. using np.repeat) to plot the piecewise constant nature of the random field...
        x = np.linspace(0.0, 1.0, 2**self.div + 1, endpoint = True).repeat(2)[1:-1]
        xs, ys = np.meshgrid(x, x)

        rpts = np.ones(self.l_values.shape[0], dtype='int')
        rpts[1:-1] = 2
        plottable = self.c_values.repeat(2, axis=0).repeat(2, axis=1) + self.l_values.repeat(rpts, axis=0).repeat(rpts, axis=1)

        if self.div > div_frame:
            wframe = ax.plot_surface(xs, ys, plottable, cstride=2**(self.div - div_frame), rstride=2**(self.div-div_frame), 
                                     cmap=cmap, alpha=alpha)
        else:
            wframe = ax.plot_surface(xs, ys, plottable, cstride=1, rstride=1, cmap=cmap, alpha=alpha)

        ax.set_axis_bgcolor('white')
        if show_axes_labels:
            ax.set_xlabel('$x$')
            ax.set_ylabel('$y$')
        if title is not None:
            ax.set_title(title)

    # Here we overload the + += - -= * and / operators
    def __add__(self, other):
        d = max(self.div,other.div)
        u = self.interpolate(d)
        v = other.interpolate(d)
        if isinstance(other, DyadicPWHybrid):
            return DyadicPWHybrid(l_values = u.l_values + v.l_values, 
                                  c_values = u.c_values + v.c_values, div=d)
        if isinstance(other, DyadicPWLinear):
            return DyadicPWHybrid(l_values = u.l_values + v.values, c_values = u.c_values, div = d)
        if isinstance(other, DyadicPWConstant):
            return DyadicPWHybrid(l_values = u.l_values, c_values = u.c_values + v.values, div = d)
        else:
            raise Exception('DyadicPWHybrid operator not suported for that type')

    __radd__ = __add__

    def __iadd__(self, other):
        d = max(self.div,other.div)
        u = self.interpolate(d)
        v = other.interpolate(d)
        if isinstance(other, DyadicPWHybrid):
            self.l_values = u.l_values + v.l_values 
            self.c_values = u.c_values + v.c_values
            self.div = d
        if isinstance(other, DyadicPWLinear):
            self.l_values = u.l_values + v.values
            self.c_values = u.c_values
            self.div = d
        if isinstance(other, DyadicPWConstant):
            self.l_values = u.l_values
            self.c_values = u.c_values + v.values
            self.div = d
        else:
            raise Exception('DyadicPWHybrid operator not suported for that type')
        return self
        
    def __sub__(self,other):
        d = max(self.div,other.div)
        u = self.interpolate(d)
        v = other.interpolate(d)
        if isinstance(other, DyadicPWHybrid):
            return DyadicPWHybrid(l_values = u.l_values - v.l_values, 
                                  c_values = u.c_values - v.c_values, div=d)
        if isinstance(other, DyadicPWLinear):
            return DyadicPWHybrid(l_values = u.l_values - v.values, c_values = u.c_values, div = d)
        if isinstance(other, DyadicPWConstant):
            return DyadicPWHybrid(l_values = u.l_values, c_values = u.c_values - v.values, div = d)
        else:
            raise Exception('DyadicPWHybrid operator not suported for that type')

    __rsub__ = __sub__

    def __isub__(self,other):   
        d = max(self.div,other.div)
        u = self.interpolate(d)
        v = other.interpolate(d)
        if isinstance(other, DyadicPWHybrid):
            self.l_values = u.l_values - v.l_values 
            self.c_values = u.c_values - v.c_values
            self.div = d
        if isinstance(other, DyadicPWLinear):
            self.l_values = u.l_values - v.values
            self.c_values = u.c_values
            self.div = d
        if isinstance(other, DyadicPWConstant):
            self.l_values = u.l_values
            self.c_values = u.c_values - v.values
            self.div = d
        else:
            raise Exception('DyadicPWHybrid operator not suported for that type')
        return self

    def __mul__(self,other):
        return DyadicPWHybrid(l_values = self.l_values * other, c_values = self.c_values * other, div = self.div)
    __rmul__ = __mul__

    def __pow__(self,power):
        raise Exception('DyadicPWHybrid: Exponentiation makes no sense')

    def __truediv__(self,other):
        return DyadicPWHybrid(l_values = self.l_values / other, c_values = self.c_values / other, div = self.div)

class Basis(object):
    """ A vaguely useful encapsulation of what you'd wanna do with a basis,
        including an orthonormalisation procedure """

    def __init__(self, vecs=None, space='H1', values_flat=None, pre_allocate=0, file_name=None):
        
        if vecs is not None:
            self.vecs = vecs
            self.n = len(vecs)
            self.space = space

            # Make a flat "values" thing for speed's sake, so we
            # can use numpy power!
            # NB we allow it to be set externally for accessing speed
            if values_flat is None:
                # Pre-allocate here is used for speed purposes... so that memory is allocated and ready to go...
                self.values_flat =  np.zeros(np.append(self.vecs[0].values.shape, max(self.n, pre_allocate)))
                for i, vec in enumerate(self.vecs):
                    self.values_flat[:,:,i] = vec.values
            else:
                if values_flat.shape[2] < self.n:
                    raise Exception('Incorrectly sized flat value matrix, are the contents correct?')
                else:
                    self.values_flat = np.zeros(np.append(self.vecs[0].values.shape, \
                                                max(self.n, pre_allocate, values_flat.shape[2]) ))
                    self.values_flat[:,:,:values_flat.shape[2]] = values_flat
                        
            self.orthonormal_basis = None
            self.G = None
            self.U = self.S = self.V = None

        elif file_name is not None:
            self.load(file_name)
        else:
            raise Exception('Basis either needs vector initialisation or file name!')
    
    def add_vector(self, vec):
        """ Add just one vector, so as to make the new Grammian calculation quick """

        self.vecs.append(vec)
        self.n += 1

        if self.G is not None:
            self.G = np.pad(self.G, ((0,1),(0,1)), 'constant')
            for i in range(self.n):
                self.G[self.n-1, i] = self.G[i, self.n-1] = self.vecs[-1].dot(self.vecs[i], space=self.space)

        self.U = self.V = self.S = None

        if self.values_flat.shape[2] <= self.n - 1:
            self.values_flat.resize((self.values_flat.shape[0], self.values_flat.shape[1], self.n))
        
        self.values_flat[:,:,self.n-1] = vec.values

    def subspace(self, indices):
        """ To be able to do "nested" spaces, the easiest way is to implement
            subspaces such that we can draw from a larger ambient space """
        return type(self)(self.vecs[indices], space=self.space, values_flat=self.values_flat[:,:,indices])

    def subspace_mask(self, mask):
        """ Here we make a subspace by using a boolean mask that needs to be of
            the same size as the number of vectors. Used for the cross validation """
        if mask.shape[0] != len(self.vecs):
            raise Exception('Subspace mask must be the same size as length of vectors')
        return type(self)(list(compress(self.vecs, mask)), space=self.space, values_flat=self.values_flat[:,:,mask])

    def dot(self, u):
        u_d = np.zeros(self.n)
        for i, v in enumerate(self.vecs):
            u_d[i] = v.dot(u, self.space)
        return u_d

    def make_grammian(self):
        if self.G is None:
            self.G = np.zeros([self.n,self.n])
            for i in range(self.n):
                for j in range(i+1):
                    self.G[i,j] = self.G[j,i] = self.vecs[i].dot(self.vecs[j], self.space)

    def cross_grammian(self, other):
        CG = np.zeros([self.n, other.n])
        
        for i in range(self.n):
            for j in range(other.n):
                CG[i,j] = self.vecs[i].dot(other.vecs[j], self.space)
        return CG

    def project(self, u):
        
        # Either we've made the orthonormal basis...
        if self.orthonormal_basis is not None:
            return self.orthonormal_basis.project(u) 
        else:
            if self.G is None:
                self.make_grammian()

            u_n = self.dot(u)
            try:
                if sparse.issparse(self.G):
                    y_n = sparse.linalg.spsolve(self.G, u_n)
                else:
                    y_n = scipy.linalg.solve(self.G, u_n, sym_pos=True)
            except np.linalg.LinAlgError as e:
                print('Warning - basis is linearly dependent with {0} vectors, projecting using SVD'.format(self.n))

                if self.U is None:
                    if sparse.issparse(self.G):
                        self.U, self.S, self.V =  scipy.sparse.linalg.svds(self.G)
                    else:
                        self.U, self.S, self.V = np.linalg.svd(self.G)
                # This is the projection on the reduced rank basis 
                y_n = self.V.T @ ((self.U.T @ u_n) / self.S)

            # We allow the projection to be of the same type 
            # Also create it from the simple broadcast and sum (which surely should
            # be equivalent to some tensor product thing??)
            #u_p = type(self.vecs[0])((y_n * self.values_flat).sum(axis=2)) 
        
            return self.reconstruct(y_n)

    def reconstruct(self, c):
        # Build a function from a vector of coefficients
        
        u_p = type(self.vecs[0])((c * self.values_flat[:,:,:self.n]).sum(axis=2)) 
        return u_p

    def matrix_multiply(self, M):
        # Build another basis from a matrix, essentially just calls 
        # reconstruct for each row in M
        if M.shape[0] != M.shape[1] or M.shape[0] != self.n:
            raise Exception('M must be a {0}x{1} square matrix'.format(self.n, self.n))

        vecs = []
        for i in range(M.shape[0]):
            vecs.append(self.reconstruct(M[i,:]))
        
        # In case this is an orthonormal basis
        return Basis(vecs, space=self.space)

    def ortho_matrix_multiply(self, M):
        # Build another basis from an orthonormal matrix, 
        # which means that the basis that comes from it
        # is also orthonormal *if* it was orthonormal to begin with
        if M.shape[0] != M.shape[1] or M.shape[0] != self.n:
            raise Exception('M must be a {0}x{1} square matrix'.format(self.n, self.n))

        vecs = []
        for i in range(M.shape[0]):
            vecs.append(self.reconstruct(M[i,:]))
        
        # In case this is an orthonormal basis
        return type(self)(vecs, space=self.space)

    def orthonormalise(self):

        if self.G is None:
            self.make_grammian()
        
        # We do a cholesky factorisation rather than a Gram Schmidt, as
        # we have a symmetric +ve definite matrix, so this is a cheap and
        # easy way to get an orthonormal basis from our previous basis
        
        if sparse.issparse(self.G):
            L = sparse.cholmod.cholesky(self.G)
        else:
            L = np.linalg.cholesky(self.G)
        L_inv = scipy.linalg.lapack.dtrtri(L.T)[0]
        
        ortho_vecs = []
        for i in range(self.n):
            ortho_vecs.append(type(self.vecs[0])((L_inv[:,i] * self.values_flat[:,:,:self.n]).sum(axis=2)))

        self.orthonormal_basis = OrthonormalBasis(ortho_vecs, self.space)

        return self.orthonormal_basis

    def save(self, file_name):
        if self.G is not None:
            if self.S is not None and self.U is not None and self.V is not None:
                np.savez_compressed(file_name, values_flat=self.values_flat, G=self.G, S=self.S, U=self.U, V=self.V)
            else:
                np.savez_compressed(file_name, values_flat=self.values_flat, G=self.G)
        else:
            np.savez_compressed(file_name, values_flat=self.values_flat)

    def load(self, file_name):

        data = np.load(file_name)

        self.values_flat = data['values_flat']
        
        self.vecs = []
        for i in range(self.values_flat.shape[-1]):
            self.vecs.append(DyadicPWLinear(self.values_flat[:,:,i]))
        
        # TODO: make this a part of the saved file format...
        self.space = 'H1'

        if 'G' in data.files:
            self.G = data['G']
        else:
            self.G = None
        if 'S' in data.files and 'U' in data.files and 'V' in data.files:
            self.S = data['S']
            self.U = data['U']
            self.V = data['V']
        else:
            self.S = self.U = self.V = None

class OrthonormalBasis(Basis):

    def __init__(self, vecs=None, space='H1', values_flat=None, file_name=None):
        # We quite naively assume that the basis we are given *is* in 
        # fact orthonormal, and don't do any testing...

        super().__init__(vecs=vecs, space=space, values_flat=values_flat, file_name=file_name)
        #self.G = np.eye(self.n)
        #self.G = sparse.identity(self.n)

    def project(self, u):
        # Now that the system is orthonormal, we don't need to solve a linear system
        # to make the projection
        y_n = self.dot(u)
        return type(self.vecs[0])((y_n * self.values_flat[:,:,:self.n]).sum(axis=2))

    def orthonormalise(self):
        return self


class BasisPair(object):
    """ This class automatically sets up the cross grammian, calculates
        beta, and can do the optimal reconstruction and calculated a favourable basis """

    def __init__(self, Wm, Vn, G=None, space='H1'):

        if Wm.space != Vn.space or Wm.space != space:
            raise Exception('Warning - all bases must have matching norms')
        if Vn.n > Wm.n:
            raise Exception('Error - Wm must be of higher dimensionality than Vn')

        self.Wm = Wm
        self.Vn = Vn
        self.m = Wm.n
        self.n = Vn.n
        self.space = space
        
        if G is not None:
            self.G = G
        else:
            self.G = self.cross_grammian()

        self.U = self.S = self.V = None

    def cross_grammian(self):
        CG = np.zeros([self.m, self.n])
        
        for i in range(self.m):
            for j in range(self.n):
                CG[i,j] = self.Wm.vecs[i].dot(self.Vn.vecs[j], self.space)
        return CG
    
    def beta(self):
        if self.U is None or self.S is None or self.V is None:
            self.calc_svd()

        return self.S[-1]

    def calc_svd(self):
        if self.U is None or self.S is None or self.V is None:
            self.U, self.S, self.V = np.linalg.svd(self.G)

    def make_favorable_basis(self):
        if isinstance(self, FavorableBasisPair):
            return self
        
        if not isinstance(self.Wm, OrthonormalBasis) or not isinstance(self.Vn, OrthonormalBasis):
            raise Exception('Both Wm and Vn must be orthonormal to calculate the favourable basis!')

        if self.U is None or self.S is None or self.V is None:
            self.calc_svd()

        fb = FavorableBasisPair(self.Wm.ortho_matrix_multiply(self.U.T), 
                                self.Vn.ortho_matrix_multiply(self.V),
                                S=self.S, U=np.eye(self.n), V=np.eye(self.m),
                                space=self.space)
        return fb

    def measure_and_reconstruct(self, u, disp_cond=False):
        """ Just a little helper function. Not sure we really want this here """ 
        u_p_W = self.Wm.dot(u)
        return self.optimal_reconstruction(u_p_W, disp_cond)

    def optimal_reconstruction(self, w, disp_cond=False):
        """ And here it is - the optimal reconstruction """
        try:
            c = scipy.linalg.solve(self.G.T @ self.G, self.G.T @ w, sym_pos=True)
        except np.linalg.LinAlgError as e:
            print('Warning - unstable v* calculation, m={0}, n={1} for Wm and Vn, returning 0 function'.format(self.Wm.n, self.Vn.n))
            c = np.zeros(self.Vn.n)

        v_star = self.Vn.reconstruct(c)

        u_star = v_star + self.Wm.reconstruct(w - self.Wm.dot(v_star))

        # Note that W.project(v_star) = W.reconsrtuct(W.dot(v_star))
        # iff W is orthonormal...
        cond = np.linalg.cond(self.G.T @ self.G)
        if disp_cond:
            print('Condition number of G.T * G = {0}'.format(cond))
        
        return u_star, v_star, self.Wm.reconstruct(w), self.Wm.reconstruct(self.Wm.dot(v_star)), cond

class FavorableBasisPair(BasisPair):
    """ This class automatically sets up the cross grammian, calculates
        beta, and can do the optimal reconstruction and calculated a favourable basis """

    def __init__(self, Wm, Vn, S=None, U=None, V=None, space='H1'):
        # We quite naively assume that the basis we are given *is* in 
        # fact orthonormal, and don't do any testing...

        if S is not None:
            # Initialise with the Grammian equal to the singular values
            super().__init__(Wm, Vn, G=S, space=space)
            self.S = S
        else:
            super().__init__(Wm, Vn, space=space)
        if U is not None:
            self.U = U
        if V is not None:
            self.V = V

    def make_favorable_basis(self):
        return self

    def optimal_reconstruction(self, w, disp_cond=False):
        """ Optimal reconstruction is much easier with the favorable basis calculated 
            NB we have to assume that w is measured in terms of our basis Wn here... """
        
        w_tail = np.zeros(w.shape)
        w_tail[self.n:] = w[self.n:]
        
        v_star = self.Vn.reconstruct(w[:self.n] / self.S)
        u_star = v_star + self.Wm.reconstruct(w_tail)

        return u_star, v_star, self.Wm.reconstruct(w), self.Wm.reconstruct(self.Wm.dot(v_star))

"""
 This code here is maintained detached from the basis pair algorithm for backwards
 compatibility with previous tests
"""
def optimal_reconstruction(W, V_n, w, disp_cond=False):
    """ And here it is - the optimal """
    BP = BasisPair(W, V_n)
    return BP.optimal_reconstruction(w, disp_cond)

"""
*****************************************************************************************
All the functions below are for building specific basis systems, reduced basis, sinusoid, 
coarse grid hat functions, etc...
*****************************************************************************************
"""

def make_hat_basis(div, space='H1'):
    # Makes a complete hat basis for division div
    V_n = []
    # n is the number of internal grid points, i.e. we will have n different hat functionsdd
    # for our coarse-grid basis
    side_n = 2**div-1

    for k in range(side_n):
        for l in range(side_n):
            v_i = DyadicPWLinear(div = div)
            v_i.values[k+1, l+1] = 1.0
            V_n.append(v_i)
    
    b = Basis(V_n, space=space)

    h = 2 ** (- b.vecs[0].div)
    # We construct the Grammian here explicitly, otherwise it takes *forever*
    # as the grammian is often used in Reisz representer calculations
    grammian = np.zeros([side_n*side_n, side_n*side_n])
    diag = (4.0 + h*h/2.0) * np.ones(side_n*side_n)
    lr_diag = (h*h/12.0 - 1) * np.ones(side_n*side_n)

    # min_diag is below the diagonal, hence deals with element to the left in the FEM grid
    lr_diag[side_n-1::side_n] = 0 # These corresponds to edges on left or right extreme
    lr_diag = lr_diag[:-1]

    ud_diag = (h*h/12.0 - 1) * np.ones(side_n*side_n)
    ud_diag = ud_diag[side_n:]
    
    grammian = sparse.diags([diag, lr_diag, lr_diag, ud_diag, ud_diag], [0, -1, 1, -side_n, side_n]).tocsr()
    b.G = grammian
     
    return b

def make_sin_basis(div, N=None, space='H1'):
    V_n = []

    if N is None:
        N = 2**div - 1

    # We want an ordering such that we get (1,1), (1,2), (2,1), (2,2), (2,3), (3,2), (3,1), (1,3), ...
    for n in range(1,N+1):
        for m in range(1,n+1):
            def f(x,y): return np.sin(n * math.pi * x) * np.sin(m * math.pi * y) * 2.0 / math.sqrt(1.0 + math.pi * math.pi * (m * m + n * n))
            v_i = DyadicPWLinear(func = f, div = div)
            V_n.append(v_i)
            
            # We do the mirrored map here
            if m < n:
                def f(x,y): return np.sin(m * math.pi * x) * np.sin(n * math.pi * y) * 2.0 / math.sqrt(1.0 + math.pi * math.pi * (m * m + n * n))

                v_i = DyadicPWLinear(func = f, div = div)
                V_n.append(v_i)

    return Basis(V_n, space=space)


def make_reduced_basis(n, field_div, fem_div, point_gen=None, space='H1', a_bar=1.0, c=0.5, f=1.0):
    # Make a basis of m solutions to the FEM problem, from random generated fields

    side_n = 2**field_div
    
    if point_gen is None:
        point_gen = pg.MonteCarlo(d=side_n*side_n, n=n, lims=[-1, 1])
    elif point_gen.n != n:
        raise Exception('Need point dictionary with right number of points!')

    V_n = []
    fields = []

    for i in range(n):
        field = DyadicPWConstant(a_bar + c * point_gen.points[i,:].reshape([side_n, side_n]), div=field_div)
        fields.append(field)
        # Then the fem solver (there a faster way to do this all at once? This will be huge...
        fem_solver = DyadicFEMSolver(div=fem_div, rand_field = field, f = 1)
        fem_solver.solve()
        V_n.append(fem_solver.u)
        
    return Basis(V_n, space=space), fields

def make_approx_basis(div, low_point=0.01, space='H1'):
    # Make it out of a few solutions FE
    side_n = 2**div
    V_n = []
    fields = []
    for i in range(side_n-1):
        for j in range(side_n-1):
            a = DyadicPWConstant(div=div)
            a.values[:,:] = 1.0
            a.values[i:i+2,j:j+2] = low_point
            fields.append(a)
            fem = DyadicFEMSolver(div=div, rand_field=a, f=1.0)
            fem.solve()
            V_n.append(fem.u)

    return Basis(V_n, space=space), fields

def make_random_local_integration_basis(m, div, width=2, bounds=None, bound_prop=1.0, space='H1', return_map=False):

    M_m = []
    
    full_points =  list(product(range(2**div - (width-1)), range(2**div - (width-1))))

    if bounds is not None:
        bound_points = list(product(range(bounds[0,0], bounds[0,1] - (width-1)), range(bounds[1,0], bounds[1,1] - (width-1))))
        remain_points = [p for p in full_points if p not in bound_points]
        remain_locs = np.random.choice(range(len(remain_points)), round(m * (1.0 - bound_prop)), replace=False)
    else:
        bound_points = full_points 
        remain_points = [] 
        remain_locs = []
        
    bound_locs = np.random.choice(range(len(bound_points)), round(m * bound_prop), replace=False)
    
    points = [bound_points[bl] for bl in bound_locs] + [remain_points[rl] for rl in remain_locs]
    
    #np.random.choice(range(len(points)), m, replace=False)
    h = 2**(-div)

    local_meas_fun = DyadicPWConstant(div=div)
    
    stencil = h*h*3.0 * np.ones([width, width])
    stencil[0,:]=stencil[-1,:]=stencil[:,0]=stencil[:,-1]=h*h*3.0/2.0
    stencil[0,0]=stencil[-1,-1]=h*h/2.0
    stencil[0,-1]=stencil[-1,0]=h*h
    
    if space == 'H1':
        hat_b = make_hat_basis(div=div, space='H1')
        hat_b.make_grammian()
    
    for i in range(m):
        point = points[i]

        local_meas_fun.values[point[0]:point[0]+width,point[1]:point[1]+width] += 1.0

        meas = DyadicPWLinear(div=div)
        meas.values[point[0]:point[0]+width,point[1]:point[1]+width] = stencil

        if space == 'H1':
            # Then we have to make this an element of coarse H1,
            # which we do by creating a hat basis and solving
            if sparse.issparse(hat_b.G):
                v = sparse.linalg.spsolve(hat_b.G, meas.values[1:-1,1:-1].flatten())
            else:
                v = scipy.linalg.solve(hat_b.G, meas.values[1:-1,1:-1].flatten(), sym_pos=True)
            meas = hat_b.reconstruct(v)
            
        M_m.append(meas)
    
    W = Basis(M_m, space)
    if return_map:
        return W, local_meas_fun

    return W


def make_local_integration_basis(div, int_div, space='H1'):

    if div < int_div:
        raise Exception('Integration div must be less than or equal to field div')

    M_m = []
    side_m = 2**int_div
    h = 2**(-div)

    int_size = 2**(div - int_div)
    stencil = h*h*3.0 * np.ones([int_size+1, int_size+1])
    stencil[0,:]=stencil[-1,:]=stencil[:,0]=stencil[:,-1]=h*h*3.0/2.0
    stencil[0,0]=stencil[-1,-1]=h*h/2.0
    stencil[0,-1]=stencil[-1,0]=h*h
   
    if space == 'H1':
        hat_b = make_hat_basis(div=div, space='H1')
        hat_b.make_grammian()

    for i in range(side_m):
        for j in range(side_m):

            meas = DyadicPWLinear(div=div)
            meas.values[i*int_size:(i+1)*int_size+1, j*int_size:(j+1)*int_size+1] = stencil

            if space == 'H1':
                # Then we have to make this an element of coarse H1,
                # which we do by creating a hat basis and solving
                if sparse.issparse(hat_b.G):
                    v = sparse.linalg.spsolve(hat_b.G, meas.values[1:-1,1:-1].flatten())
                else:
                    v = scipy.linalg.solve(hat_b.G, meas.values[1:-1,1:-1].flatten(), sym_pos=True)
                meas = hat_b.reconstruct(v)

            M_m.append(meas)
    
    W = Basis(M_m, space)
    return W

"""
*****************************************************************************************
All the functions below are for building bases from greedy algorithms. Several
variants are proposed here.
*****************************************************************************************
"""

def make_dictionary(point_gen, fem_div, a_bar=1.0, c=0.5, f=1, verbose=False):
    
    basis_div = int(round(math.log(math.sqrt(point_gen.d), 2)))
    # First we need a point process generator
    if point_gen.d != 2**basis_div * 2**basis_div:
        raise Exception('Point generator is not of a correct dyadic level dimension')

    # Now we feed that in to the field generatr
    D = []
    fields = []
    
    if verbose:
        print('Generating dictionary point: ', end='')
    
    for i in range(point_gen.n):
        field = DyadicPWConstant(a_bar + c * point_gen.points[i,:].reshape([2**basis_div,2**basis_div]), div=basis_div)
        fields.append(field)
        # Then the fem solver (there a faster way to do this all at once? This will be huge...
        fem_solver = DyadicFEMSolver(div=fem_div, rand_field=field, f=f)
        fem_solver.solve()
        D.append(fem_solver.u)

        if verbose and i % 100 == 0:
            print('{0}... '.format(i), end='')

    if verbose: 
        print('')

    return D, fields

class GreedyBasisConstructor(object):
    """ This is the original greedy algorithm that minimises the Kolmogorov n-width, and a 
        generic base-class for all other greedy algorithms """

    def __init__(self, n, fem_div, dictionary=None, point_gen=None, a_bar=1.0, c=0.5, verbose=False, remove=True):
        """ We need to be either given a dictionary or a point generator that produces d-dimensional points
            from which we generate the dictionary. """
        
        if point_gen is not None and dictionary is not None:
            raise Exception('Both point generator and dictionary are defined... only one is to be used though, which?')
        elif point_gen is not None:
            self.dictionary, self.fields = make_dictionary(point_gen, fem_div, a_bar, c, verbose=verbose)
        elif dictionary is not None:
            # We make a shallow local copy so that we can remove elements from the list...
            self.dictionary = copy.copy(dictionary)
        else:
            raise Exception('Must specify either point generator or dicitonary')

        self.n = n
        self.verbose = verbose
        self.remove = remove
        self.greedy_basis = None

    def initial_choice(self):
        """ Different greedy methods will have their own maximising/minimising criteria, so all 
        inheritors of this class are expected to overwrite this method to suit their needs. """

        self.norms = np.zeros(len(self.dictionary))
        for i in range(len(self.dictionary)):
            self.norms[i] = self.dictionary[i].norm(space='H1')
        
        n0 = np.argmax(self.norms)
        if self.remove:
            self.norms = np.delete(self.norms, n0)

        return n0

    def next_step_choice(self, i):
        """ Different greedy methods will have their own maximising/minimising criteria, so all 
        inheritors of this class are expected to overwrite this method to suit their needs. """

        p_V_d = np.zeros(len(self.dictionary))
        # We go through the dictionary and find the max of || f ||^2 - || P_Vn f ||^2
        for j in range(len(self.dictionary)):
            p_V_d[j] = self.greedy_basis.project(self.dictionary[j]).norm('H1')
        
        next_crit = self.norms * self.norms - p_V_d * p_V_d
        ni = np.argmax(next_crit)

        if self.remove:
            self.norms = np.delete(self.norms, ni)

        if self.verbose:
            print('{0} : \t {1} \t {2}'.format(i, self.norms[ni], next_crit[ni]))

        return ni

    def construct_basis(self):
        " The construction method should be generic enough to support all variants of the greedy algorithms """
        
        if self.greedy_basis is None:
            n0 = self.initial_choice()

            self.greedy_basis = Basis([self.dictionary[n0]], space='H1', pre_allocate=self.n)
            self.greedy_basis.make_grammian()
            
            if self.remove:
                del self.dictionary[n0]

            if self.verbose:
                print('\n\nGenerating basis from greedy algorithm with dictionary: ')
                print('i \t || phi_i || \t\t || phi_i - P_V_(i-1) phi_i ||')

            for i in range(1, self.n):
                
                ni = self.next_step_choice(i)
                    
                self.greedy_basis.add_vector(self.dictionary[ni])
                if self.remove:
                    del self.dictionary[ni]
                       
            if self.verbose:
                print('\n\nDone!')
        else:
            print('Greedy basis already computed!')

        return self.greedy_basis

class MBGreedyBasisConstructor(GreedyBasisConstructor):
    """ This greedy algorithm performs the same perpendicular maximisation, but projected on the
        measurements space provided. This was an attempt to ammeliorate the beta condition between the
        approx space Vn and meas. space Wm.... but didn't work too well """

    def __init__(self, n, fem_div, Wm, dictionary=None, point_gen=None, a_bar=1.0, c=0.5, verbose=False, remove=True):
        """ We need to be either given a dictionary or a point generator that produces d-dimensional points
            from which we generate the dictionary. """
        
        super().__init__(n, fem_div, dictionary=dictionary, point_gen=point_gen, a_bar=a_bar, c=c, verbose=verbose, remove=remove)
        
        self.Wm = Wm
        self.dots = None

    def initial_choice(self):
        """ The initial choice here is the vector in the dictionary that is maximum
            in the Wm projection, i.e. we dot prod the dictionary members against Wm and look for
            the maximum. For efficiency means we start constructing Vn_W, the projection of the
            Vn basis that we are constructing projected in Wm... """

        self.dots = np.zeros((len(self.dictionary), self.Wm.n))
        self.norms = np.zeros(len(self.dictionary))
        for i in range(len(self.dictionary)):
            self.dots[i,:] = self.Wm.dot(self.dictionary[i])
            self.norms[i] = np.linalg.norm(self.dots[i])
        
        n0 = np.argmax(self.norms)

        # And this is the greedy basis in projection space...
        self.Vn_W = np.zeros([self.n, self.Wm.n])
        self.Vn_W[0,:] = self.dots[n0, :]

        self.Vn_loc_G = np.zeros([self.n,self.n])
        self.Vn_loc_G[0,0] = np.dot(self.Vn_W[0,:], self.Vn_W[0,:])

        if self.remove:
            self.norms = np.delete(self.norms, n0)
            self.dots = np.delete(self.dots, n0, 0)

        return n0

    def next_step_choice(self, i):
        """ Here we chose the member of the dictionary that has the largest perindicular distance, but
            projected in the Wm space, so we need to calculate the projection of each dictionary
            member in Vn projected to Wm... """

        G = self.Vn_loc_G[:i,:i]
        p_V_d = np.zeros(len(self.dictionary))
        for j in range(len(self.dictionary)):

            c = scipy.linalg.solve(G, np.dot(self.Vn_W[:i,:], self.dots[j]), sym_pos=True)
            p_v_Vn_Wm = np.dot(c, self.Vn_W[:i])
            p_V_d[j] = np.linalg.norm(self.dots[j] - p_v_Vn_Wm)

        ni = np.argmax(p_V_d)

        if self.verbose:
            print('{0} : \t {1} \t {2} \t {3}'.format(i, ni, self.norms[ni], p_V_d[ni]))

        self.Vn_W[i, :] = self.dots[ni]
        for j in range(i):
            self.Vn_loc_G[i,j] = self.Vn_loc_G[j,i] = np.dot(self.Vn_W[i,:], self.Vn_W[j,:])
            self.Vn_loc_G[i,i] = np.dot(self.Vn_W[i,:], self.Vn_W[i,:])
        
        if self.remove:
            self.norms = np.delete(self.norms, ni)
            self.dots = np.delete(self.dots, ni, 0)

        return ni 

class DBGreedyBasisConstructor(GreedyBasisConstructor):
    """ This is the first of the two proposed data-based greedy basis constructors """

    def __init__(self, n, fem_div, Wm, w, dictionary=None, point_gen=None, verbose=False):
        """ We need to be either given a dictionary or a point generator that produces d-dimensional points
            from which we generate the dictionary. """
        
        super().__init__(n, fem_div, dictionary=dictionary, point_gen=point_gen, verbose=verbose)
        
        self.Wm = Wm
        self.w = w
        self.w_r = self.Wm.reconstruct(w)

        # This is the dictionary projected in to Wm 
        self.D_Wm = None

    def initial_choice(self):

        self.D_Wm = []
        crit = np.zeros(len(self.dictionary))
        for i in range(len(self.dictionary)):
            dot = self.Wm.dot(self.dictionary[i])

            self.D_Wm.append(self.Wm.reconstruct(dot) / np.linalg.norm(dot))
            crit[i] = np.abs(self.w_r.dot(self.D_Wm[-1]))
        
        n0 = np.argmax(crit)

        if self.remove:
            del self.D_Wm[n0]

        return n0

    def next_step_choice(self, i):
        
        perp_data_dist = self.w_r - self.greedy_basis.project(self.w_r)
        crit = np.zeros(len(self.dictionary))
        for j in range(len(self.D_Wm)):
            crit[j] = np.abs(perp_data_dist.dot(self.D_Wm[j]))

        ni = np.argmax(crit)

        if self.verbose:
            print('{0} : \t {1} \t {2} \t {3}'.format(i, ni, crit[ni], perp_data_dist.norm()))

        if self.remove:
            del self.D_Wm[ni]

        return ni 

class PPGreedyBasisConstructor(GreedyBasisConstructor):
    """ This is the first of the two proposed data-based greedy basis constructors """

    def __init__(self, n, fem_div, Wm, w, dictionary=None, point_gen=None, verbose=False):
        """ We need to be either given a dictionary or a point generator that produces d-dimensional points
            from which we generate the dictionary. """
        
        super().__init__(n, fem_div, dictionary=dictionary, point_gen=point_gen, verbose=verbose)
        
        self.Wm = Wm
        self.w = w
        self.w_r = self.Wm.reconstruct(w)

    def initial_choice(self):
        
        dots = np.zeros(len(self.dictionary))
        for i in range(len(self.dictionary)):
            dots[i] = np.abs(self.dictionary[i].dot(self.w_r, space='H1'))
    
        n0 = np.argmax(dots)
        
        if self.verbose:
            print('{0} : \t {1} \t {2}'.format(0, n0, dots[n0]))

        return n0

    def next_step_choice(self, i):
        
        dots = np.zeros(len(self.dictionary))
        for j in range(len(self.dictionary)):
           
            perp_data_dist = self.dictionary[j] - self.greedy_basis.project(self.dictionary[j])
            dots[j] = np.abs(perp_data_dist.dot(self.w_r) / perp_data_dist.norm(space='H1'))
        
        ni = np.argmax(dots)

        if self.verbose:
            print('{0} : \t {1} \t {2}'.format(i, ni, dots[ni]))

        return ni 

def greedy_reduced_basis_construction(n, field_div, fem_div, point_gen=None, dictionary=None, a_bar=1.0, c=0.5, verbose=False):
    """ This is the "vanilla" flavoured greedy basis algorithm that minimises the Kolmogorov n-width 
        n is the final size of the basis 
        
        NB This function is maintained purely for backwards compatibility with old tests. It now links to new code
        contained in the GreedyBasisConstructor class """
    
    Vn_const = grb_cons.construct_basis()
    Vn_greedy = greedy_reduced_basis_construction(n=m, field_div=field_div, fem_div=fem_div, \
                                                  point_gen=point_gen, a_bar=a_bar, c=c, verbose=True)
    return Vn_greedy

def measurement_based_greedy_reduced_basis_construction(n, field_div, fem_div, point_gen, Wm, a_bar=1.0, c=0.5, verbose=False):
    """ Here we apply the greedy algorithm but with respect to the measurement space 
        Note that we assume Wm to be orthonormal! """

    if not isinstance(Wm, OrthonormalBasis):
        raise Exception('Wm must be an orthonormal basis for the data-based greedy algorithm')
    
    Vn_cons = MBGreedyBasisConstructor(n=m, fem_div=fem_div, Wm=Wm, point_gen=point_gen, a_bar=a_bar, c=c, verbose=True)
    Vn_greedy = mrb_cons.construct_basis()

    return Vn_greedy
    
