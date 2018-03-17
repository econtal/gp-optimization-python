"""
This file is part of GpOptimization.

GpOptimization is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, version 3 of the License.

GpOptimization is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with GpOptimization. If not, see <http://www.gnu.org/licenses/>.

Copyright (c) by Emile Contal, 2016
"""

import sys
import numpy
from scipy import linalg
from scipy.optimize import minimize

from posterior import Posterior
from kernel import KernelSE
from basis import BasisQuad


def optimize(gp, Xt, Yt, HPini=None, nelderMeadIters=50):
    if gp.KernelFun is None:
        gp.KernelFun = KernelSE
    KernelFun = gp.KernelFun
    if HPini is None:
        HPiniK = KernelFun.defaultHP(Xt)
        HPini = numpy.log(numpy.r_[Yt.var(), 1e-2, HPiniK])
    if gp.Basis is None:
        gp.Basis = BasisQuad()
    Basis = gp.Basis

    if Basis is not None:
        Ht = Basis(Xt)
    else:
        Ht = None

    f = lambda hp: nllcost(gp, Xt, Yt, hp[0], hp[1], hp[2:], Ht)

    # Starts with few Nelder-Mead iterations
    res = minimize(f, HPini, method='Nelder-Mead',
                   options={'maxiter': nelderMeadIters, 'disp': False})
    hpopt = res.x
    # Search
    res = minimize(f, hpopt, method='SLSQP', options={'disp': False, })
    HP = res.x
    kernel = KernelFun(numpy.exp(HP[0]), numpy.exp(HP[2:]))
    noise = numpy.exp(HP[0] + HP[1])
    Ktt = kernel(Xt, Xt)
    post = Posterior(Ktt, Yt, noise, Ht)
    return post, kernel, HP, noise, res.fun


def nllcost(gp, Xt, Yt, log_sf2, log_rsn, log_W, Ht):
    sf2 = numpy.exp(log_sf2)
    noise = numpy.exp(log_sf2 + log_rsn)
    W = numpy.exp(log_W)
    if not numpy.isfinite(numpy.r_[sf2, noise, W]).all():
        return numpy.inf

    kernel = gp.KernelFun(sf2, W)
    Ktt = kernel(Xt, Xt)
    if not numpy.isfinite(Ktt).all():
        return numpy.inf

    post = Posterior(Ktt, Yt, noise, Ht)
    gp.post = post

    return gp._loolik(Ktt, Yt, Ht)
