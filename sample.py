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

import numpy

from kernel import KernelSEnormiso
from chol import cholpsd


def sample(d=1, size=40, ns=1000, nt=10, kernel=None, basis=None, noise=1e-2):
    if kernel is None:
        kernel = KernelSEnormiso(d)
    Xs = size * numpy.random.rand(ns, d)
    Xt = Xs[:nt, :]
    Kss = kernel(Xs, Xs)
    Fs = numpy.dot(cholpsd(Kss).T, numpy.random.randn(ns, 1))
    if basis is not None:
        B = basis(Xs)
        Fs = Fs + numpy.dot(B, numpy.randomrandn(B.shape[1], 1))

    f = lambda X: Fs[X] + noise * numpy.random.randn(X.shape[0], 1)
    Yt = f(numpy.arange(nt))
    return f, Xs, Fs, Xt, Yt, Kss
