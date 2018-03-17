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
from scipy.spatial import distance


class Kernel(object):
    maxSpace = numpy.inf

    def __call__(self, X1, X2=None):
        if X2 is None:
            return self.diag(X1)
        else:
            self._checkInputs(X1, X2)
            return self.pairwise(X1, X2)

    def _checkInputs(self, X1, X2):
        n1, d1 = X1.shape
        n2, d2 = X2.shape
        assert(d1 == d2)
        if n1 * n2 * d1 > self.maxSpace:
            raise NotImplementedError('Matrix too big for this kernel ({},{}) ({},{})'.format(n1,d1,n2,d2))

class KernelSE(Kernel):
    maxSpace = 2**30

    def __init__(self, s, ells):
        self.s = s
        self.ARD = numpy.diag(ells, 0)

    def pairwise(self, X1, X2):
        D = sq_dist(numpy.dot(X1, self.ARD), numpy.dot(X2, self.ARD))
        return self.s * numpy.exp(-D / 2)

    def diag(self, X):
        return self.s * numpy.ones((X.shape[0], 1))

    @staticmethod
    def defaultHP(X):
        var = X.var(axis=0)
        var[var<1e-20] = 1.
        return var


class KernelSEnormiso(KernelSE):

    def __init__(self, d):
        self.s = 1
        self.ARD = numpy.ones((d, d))


def sq_dist(X1, X2):
    return distance.cdist(X1, X2, 'sqeuclidean')
