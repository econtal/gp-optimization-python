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


class Basis(object):
    pass


class BasisCst(Basis):

    def __call__(self, X):
        return numpy.ones((X.shape[0], 1))


class BasisLin(Basis):

    def __call__(self, X):
        return numpy.c_[numpy.ones((X.shape[0],)), X]


class BasisQuad(Basis):

    def __call__(self, X):
        return numpy.c_[numpy.ones((X.shape[0],)), X, X**2]
