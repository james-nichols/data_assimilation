"""
make_greedy_bases.py

Author: James Ashton Nichols
Start date: March 2017 

This is a simple script to create and save various greedy bases
"""

import numpy as np
import importlib
import dyadic_fem as df
import point_generator as pg
import seaborn as sns
import matplotlib.pyplot as plt
import pdb
importlib.reload(df)
importlib.reload(pg)
%matplotlib inline

fem_div = 7

a_bar = 1.0
c = 0.9

np.random.seed(1)
y_4 = 2 * np.random.random(2**1 * 2**1) - 1
a_4 = df.DyadicPWConstant(a_bar + c * y_4.reshape([2**1, 2**1]), div=1)
fem_4 = df.DyadicFEMSolver(div=fem_div, rand_field=a_4, f=1.0)
fem_4.solve()

np.random.seed(1)
y_16 = 2 * np.random.random(2**2 * 2**2) - 1
a_16 = df.DyadicPWConstant(a_bar + c * y_16.reshape([2**2, 2**2]), div=2)
fem_16 = df.DyadicFEMSolver(div=fem_div, rand_field=a_16, f=1.0)
fem_16.solve()

np.random.seed(1)
y_64 = 2 * np.random.random(2**3 * 2**3) - 1
a_64 = df.DyadicPWConstant(a_bar + c * y_64.reshape([2**3, 2**3]), div=3)
fem_64 = df.DyadicFEMSolver(div=fem_div, rand_field=a_64, f=1.0)
fem_64.solve()

# local_width is the width of the measurement squares
local_width = 2

m = 120

# We make the ambient spaces for Wm and Vn
np.random.seed(2)

# we make a bounding box to be the quarter square
side_n = 2**fem_div-1
bounding_box = np.array([[0, 2**(fem_div-1)], [0, 2**(fem_div-1)]])

Wm, Wloc = df.make_random_local_integration_basis(m=m, div=fem_div, width=local_width, space='H1', return_map=True)
Wm = Wm.orthonormalise()

field_div = 2

# This is the measurement vector
w = Wm.dot(fem_16.u)

Vn_sin = df.make_sin_basis(fem_div, N=2**(fem_div-2), space='H1')
Vn_red, Vn_red_fields = df.make_reduced_basis(n=m, field_div=field_div, fem_div=fem_div, space='H1', a_bar=a_bar, c=c)

point_gen_qmc = pg.QMCLatticeRule(d=2**field_div*2**field_div, n=5000, lims=[-1, 1])

point_gen_l_qmc = pg.QMCLatticeRule(d=2**field_div*2**field_div, n=50000, lims=[-1, 1])

u_dict_qmc, dict_fields_qmc = df.make_dictionary(point_gen_qmc, fem_div, a_bar=a_bar, c=c, verbose=False)

u_dict_l_qmc, dict_fields_l_qmc = df.make_dictionary(point_gen_l_qmc, fem_div, a_bar=a_bar, c=c, verbose=False)

db_cons_qmc = df.DBGreedyBasisConstructor(n=m, fem_div=fem_div, Wm=Wm, w=w, dictionary=u_dict_qmc, verbose=True)
Vn_db_qmc = db_cons_qmc.construct_basis()
Vn_db_qmc.save('Vn_db_qmc')

db_cons_l_qmc = df.DBGreedyBasisConstructor(n=m, fem_div=fem_div, Wm=Wm, w=w, dictionary=u_dict_l_qmc, verbose=True)
Vn_db_l_qmc = db_cons_l_qmc.construct_basis()
Vn_db_l_qmc.save('Vn_db_l_qmc')

pp_cons_qmc = df.PPGreedyBasisConstructor(n=m, fem_div=fem_div, Wm=Wm, w=w, dictionary=u_dict_qmc, verbose=True)
Vn_pp_qmc = pp_cons_qmc.construct_basis()
Vn_pp_qmc.save('Vn_pp_qmc')

pp_cons_l_qmc = df.PPGreedyBasisConstructor(n=m, fem_div=fem_div, Wm=Wm, w=w, dictionary=u_dict_l_qmc, verbose=True)
Vn_pp_l_qmc = pp_cons_l_qmc.construct_basis()
Vn_pp_l_qmc.save('Vn_pp_l_qmc')
w = Wm.dot(fem_16.u)
u_p_w = Wm.project(fem_16.u)

ns = range(2, m+1, 2)

bases = [Vn_db_qmc, Vn_db_l_qmc, Vn_pp_qmc, Vn_pp_l_qmc]

stats = np.zeros([6, len(bases), len(ns)])

for i, n in enumerate(ns):

    for j, Vn_big in enumerate(bases):
    
        Vn = Vn_big.subspace(slice(0,n))

        u_p_v = Vn.project(fem_16.u)
        BP = df.BasisPair(Wm, Vn)
        FB = BP.make_favorable_basis()
        u_star, v_star, w_p, v_w_p = FB.measure_and_reconstruct(fem_16.u)

        stats[0, j, i] = (fem_16.u-u_star).norm(space='H1')
        stats[1, j, i] = (u_star - v_star).norm(space='H1')
        stats[2, j, i] = FB.beta()
        stats[3, j, i] = np.linalg.cond(BP.G.T @ BP.G)
        stats[4, j, i] = (fem_16.u - u_p_v).norm(space='H1')

np.save('qmc_stats', stats)
