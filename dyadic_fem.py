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

        if seed is not None:
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
        
        if (values.shape[0] != values.shape[1] or values.shape[0] != 2**div):
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

    def dot(self, f, space='L2'):
        if space == 'L2':
            return self.L2_dot(f)
        elif space == 'H1':
            return self.H1_dot(f)
        else:
            raise Exception('Unrecognised Hilbert space norm ' + space)

    def norm(self, space='L2'):
        return self.dot(self, space)
    
    def H1_dot(self, f):
        """ Compute the H1_0 dot product with another DyadicPWLinear function
            automatically interpolates the coarser function """

        u, v, match_div = self.match_grids(f)

        h = 2.0**(-match_div)
        n_side = 2**match_div

        # This is du/dy
        p = 2 * np.ones([n_side, n_side+1])
        p[:,0] = p[:,-1] = 1
        dot = (p * (u[:-1,:] - u[1:,:]) * (v[:-1,:] - v[1:,:])).sum()
        # And this is du/dx
        p = 2 * np.ones([n_side+1, n_side])
        p[0,:] = p[-1,:] = 1
        dot = dot + (p * (u[:,1:] - u[:,:-1]) * (v[:,1:] - v[:,:-1])).sum()
        
        return 0.5 * dot + self.L2_inner(u,v,h)

    def L2_dot(self, f):
        """ Compute the L2 dot product with another DyadicPWLinear function,
            automatically interpolates the coarser function """
        
        u, v, match_div = self.match_grids(f)

        h = 2.0**(-match_div)

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
            u,v,d = self.match_grids(other)
            return DyadicPWLinear(u + v, d)
        else:
            return DyadicPWLinear(self.values * other, self.div)

    __radd__ = __add__

    def __iadd__(self,other):
        if isinstance(other, DyadicPWLinear):
            u,v,d = self.match_grids(other)
            self.div = d
            self.values = u + v
        else:
            self.values = self.values + other
        return self
        
    def __sub__(self,other):
        if isinstance(other, DyadicPWLinear):
            u,v,d = self.match_grids(other)
            return DyadicPWLinear(u - v, d)
        else:
            return DyadicPWLinear(self.values * other, self.div)
    __rsub__ = __sub__

    def __isub__(self,other):
        if isinstance(other, DyadicPWLinear):
            u,v,d = self.match_grids(other)
            self.div = d
            self.values = u - v
        else:
            self.values = self.values - other
        return self

    def __mul__(self,other):
        if isinstance(other, DyadicPWLinear):
            u,v,d = self.match_grids(other)
            return DyadicPWLinear(u * v, d)
        else:
            return DyadicPWLinear(self.values * other, self.div)
    __rmul__ = __mul__

    def __pow__(self,power):
        return DyadicPWLinear(self.values**power, self.div)

    def __truediv__(self,other):
        if isinstance(other, DyadicPWLinear):
            u,v,d = self.match_grids(other)
            return DyadicPWLinear(u / v, d)
        else:
            return DyadicPWLinear(self.values / other, self.div)

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
        # NB we can overide the norm to get different projections if ew want
        u_d = np.zeros(self.n)
        for i, v in enumerate(self.vecs):
            u_d[i] = v.dot(u, self.space)
        return u_d

    def make_grammian(self):
        self.G = np.zeros([self.n,self.n])
        for i in range(self.n):
            for j in range(i+1):
                self.G[i,j] = self.G[j,i] = self.vecs[i].dot(self.vecs[j], self.space)

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
            u_p = type(self.vecs[0])((y_n * self.values_flat).sum(axis=2)) 
        
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

def make_sine_basis(div, N, M, space='L2'):
    V_n = []
    # n is the number of internal grid points, i.e. we will have n different hat functionsdd
    # for our coarse-grid basis

    for n in range(1,N+1):
        for m in range(1,M+1):
            def f(x,y): return np.sin(n * math.pi * x) * np.sin(m * math.pi * y)
            
            v_i = DyadicPWLinear(func = f, div = div)
            V_n.append(v_i)

    return Basis(V_n, space=space)

def make_random_approx_basis(N, fem_div, field_div, space='L2', a_bar=1.0, c=0.5, seed=None):
    # Make a basis of N solutions to the FEM problem, from random generated fields

    if (field_div > fem_div):
        raise Exception('Dyadic subdivision for FEM solution must be larger than for the random field')

    V_n = []
    fields = []
    for n in range(N):
        a = DyadicRandomField(div=field_div, a_bar=a_bar, c=c, seed=seed)
        fields.append(a)
        fem = DyadicFEMSolver(div=fem_div, rand_field=a, f=1.0)
        fem.solve()
        V_n.append(fem.u)

    return Basis(V_n, space=space), fields


def make_approx_basis(div, field_div):
    # Make it out of a few solutions FEM
    pass


class Measurements(object):
    """ A measurement of the solution u of the PDE / FEM solution, in some linear subspace W """

class RandomPointMeasurements(Measurements):

    def __init__(self, i, j):
        self.i = i
        self.j = j

