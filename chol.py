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
Copyright (c) by Emile Contal, 2015
"""

import numpy
from scipy import linalg


def cholpsd(X):
    m = numpy.abs(X).min()
    m = max(numpy.finfo(float).eps, m * 1e-14)
    e = m
    I = numpy.eye(X.shape[0])
    ok = False
    while not ok:
        try:
            R = linalg.cholesky(X)
            ok = True
        except linalg.LinAlgError:
            # if the Cholesky decomposition failed, try to add a small epsilon
            # on the diagonal
            X = X + e * I
            if e > 1e6 * m:
                print('Warning, adding {} for cholpsd'.format(e))
            e = 10 * e
        except ValueError, e:
            print str(e)
            import pdb
            pdb.set_trace()
    return R


def solve_chol(R, Y):
    return linalg.cho_solve((R, False), Y)
