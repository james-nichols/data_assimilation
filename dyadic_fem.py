import math
import numpy as np
import scipy as sp
import scipy.signal
import scipy.special
import scipy.optimize
import scipy.interpolate
from scipy import sparse
from itertools import *
import inspect

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import cm 
from mpl_toolkits.mplot3d import axes3d, Axes3D

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

    def dot(self, other, space='L2'):
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

    def dot(self, other, space='L2'):
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

    def norm(self, space='L2'):
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

    def interpolate(self, interp_div):
        """ Simple interpolation routine to make this function on a finer division dyadic grid """
        
        if interp_div < self.div:
            raise Exception("Interpolation division smaller than field division! Need to integrate")
        elif interp_dif == self.div:
            return self
        else:
            const = self.c_values.repeat(2**(div-self.div), axis=0).repeat(2**(div-self.div), axis=1)
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

    def __init__(self, vecs, space='L2'):
        # No smart initialisation here...
        self.vecs = vecs
        self.n = len(vecs)
        self.space = space
        # Make a flat "values" thing for speed's sake, so we
        # can use numpy power!
        self.values_flat = self.vecs[0].values
        for vec in self.vecs[1:]:
            self.values_flat = np.dstack((self.values_flat, vec.values))

        self.orthonormal_basis = None
        self.G = None

    def dot(self, u):
        u_d = np.zeros(self.n)
        for i, v in enumerate(self.vecs):
            u_d[i] = v.dot(u, self.space)
        return u_d

    def make_grammian(self):
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
            y_n = np.linalg.solve(self.G, u_n)

            # We allow the projection to be of the same type 
            # Also create it from the simple broadcast and sum (which surely should
            # be equivalent to some tensor product thing??)
            #u_p = type(self.vecs[0])((y_n * self.values_flat).sum(axis=2)) 
        
            return self.reconstruct(y_n)

    def reconstruct(self, c):
        # Build a function from a vector of coefficients
        
        u_p = type(self.vecs[0])((c * self.values_flat).sum(axis=2)) 
        return u_p

    def orthonormalise(self):

        if self.G is None:
            self.make_grammian()
        
        # We do a cholesky factorisation rather than a Gram Schmidt, as
        # we have a symmetric +ve definite matrix, so this is a cheap and
        # easy way to get an orthonormal basis from our previous basis
        L = np.linalg.cholesky(self.G)
        L_inv = scipy.linalg.lapack.dtrtri(L.T)[0]
        
        ortho_vecs = []
        for i in range(self.n):
            ortho_vecs.append(type(self.vecs[0])((L_inv[:,i] * self.values_flat).sum(axis=2)))

        self.orthonormal_basis = OrthonormalBasis(ortho_vecs, self.space)

        return self.orthonormal_basis

class OrthonormalBasis(Basis):

    def __init__(self, vecs, space='L2'):
        super().__init__(vecs, space)
        self.G = np.eye(self.n)

    def project(self, u):
        # Now that the system is orthonormal, we don't need to solve a linear system
        # to make the projection
        y_n = self.dot(u)
        return type(self.vecs[0])((y_n * self.values_flat).sum(axis=2))

    def orthonormalise(self):
        return self

def make_hat_basis(div, space='L2'):
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

    return Basis(V_n, space=space)

def make_sine_basis(div, N=None, M=None, space='L2'):
    V_n = []

    if N is None:
        N = 2**div - 1
    if M is None:
        M = 2**div - 1
    # n is the number of internal grid points, i.e. we will have n different hat functionsdd
    # for our coarse-grid basis

    for n in range(1,N+1):
        for m in range(1,M+1):
            def f(x,y): return np.sin(n * math.pi * x) * np.sin(m * math.pi * y)
            
            v_i = DyadicPWLinear(func = f, div = div)
            V_n.append(v_i)

    return Basis(V_n, space=space)

def make_random_approx_basis(m, div, width=2, space='L2', a_bar=1.0, c=0.5, seed=None):
    # Make a basis of m solutions to the FEM problem, from random generated fields

    V_n = []
    fields = []

    points = list(product(range(2**div - (width-1)), range(2**div - (width-1))))
    locs = np.random.choice(range(len(points)), m, replace=False)

    for i in range(m):
        #a = make_dyadic_random_field(div=div, a_bar=a_bar, c=c, seed=seed)
        a = DyadicPWConstant(div=div)
        a.values[:,:] = a_bar 

        point = points[locs[i]]
        a.values[point[0]:point[0]+width, point[1]:point[1]+width] = a_bar - c
        fields.append(a)
        fem = DyadicFEMSolver(div=div, rand_field=a, f=1.0)
        fem.solve()
        V_n.append(fem.u)

    return Basis(V_n, space=space), fields


def make_approx_basis(div, low_point=0.01, space='L2'):
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

def make_random_local_measurements_basis(m, div, width=2):

    M_m = []
    
    points = list(product(range(2**div - (width-1)), range(2**div - (width-1))))
    locs = np.random.choice(range(len(points)), m, replace=False)
    h = 2**(-div)
    for i in range(m):
        point = points[locs[i]]
        meas = DyadicPWConstant(div=div)
        meas.values[point[0]:point[0]+width,point[1]:point[1]+width] = 1.0 / (width*width*h*h)
            
        M_m.append(meas)
    W = Basis(M_m, 'H1')
    return W

def optimal_reconstruction(W, V_n, w, disp_cond=False):
    """ And here it is - the optimal """
    G = W.cross_grammian(V_n)
    #w = W.dot(u)
    if disp_cond:
        print('Condition number of G.T * G = {0}'.format(np.linalg.cond(G.T @ G)))
    c = np.linalg.solve(G.T @ G, G.T @ w)

    v_star = V_n.reconstruct(c)

    u_star = v_star + W.reconstruct(w - W.dot(v_star))

    # Note that W.project(v_star) = W.reconsrtuct(W.dot(v_star))
    # iff W is orthonormal...
    return u_star, v_star, W.reconstruct(w), W.reconstruct(W.dot(v_star))


