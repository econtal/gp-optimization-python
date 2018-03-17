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
from chol import solve_chol, cholpsd

class Posterior(object):

    def __init__(self, Ktt, Yt, noise, Ht=None):

        C = Ktt + noise * numpy.eye(Ktt.shape[0])  # (nt x nt)

        # Cholesky decomposition
        self.RC = cholpsd(C)  # (nt x nt)
        if Ht is not None:
            HCH = numpy.dot(Ht.T, solve_chol(self.RC, Ht))
            self.RHCH = cholpsd(HCH)  # (b x b)

            # system resolution
            HRY = numpy.dot(Ht.T, solve_chol(self.RC, Yt))
            self.bet = solve_chol(self.RHCH, HRY)  # (b x 1)
            self.invCY = solve_chol(self.RC, Yt - numpy.dot(Ht, self.bet))
        else:
            self.invCY = solve_chol(self.RC, Yt)
            self.RHCH = None
            self.bet = None
