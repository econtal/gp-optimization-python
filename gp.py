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
from scipy import linalg
from choldate import cholupdate

from posterior import Posterior
from chol import solve_chol, cholpsd
import prior


class GP(object):
    def __init__(self, KernelFun=None, Basis=None):
        self.KernelFun = KernelFun
        self.Basis = Basis
        self.post = None
        self.kernel = None
        self.noise = None
        self.X = None

    def fit(self, X, Y):
        if self.kernel is None or self.Basis is None:
            raise RuntimeError('you should call autoFit before')
        K = self.kernel(X, X)
        H = self.Basis(X)
        self.post = Posterior(K, Y, self.noise, H)

    def autoFit(self, X, Y):
        post, kernel, _, noise, _ = prior.optimize(self, X, Y, nelderMeadIters=20)
        self.post = post
        self.kernel = kernel
        self.noise = noise
        self.X = X

    def predict(self, Xs):
        if self.kernel is None or self.Basis is None:
            raise RuntimeError('you should call autoFit before')
        Kts = self.kernel(self.X, Xs)
        dKss = self.kernel(Xs)
        H = self.Basis(self.X)
        Hs = self.Basis(Xs)
        mu, s2 = self._pred(Kts, dKss, H, Hs)
        return mu, s2

    def _pred(self, Kts, dKss, Ht=None, Hs=None, computeSigma2=True):
        if self.post is None:
            raise RuntimeError('you should call fit or autoFit before')

        post = self.post

        if Ht is not None:
            mu = numpy.dot(Hs, post.bet) + \
                numpy.dot(Kts.T, post.invCY)  # (ns x 1)
        else:
            mu = numpy.dot(Kts.T, post.invCY)
        mu = mu.flatten()

        # sigma2
        if computeSigma2:
            Vf = linalg.solve(post.RC.T, Kts)  # (nt x ns)
            covf = dKss - (Vf * Vf).sum(axis=0).reshape(-1, 1)

            if Ht is not None:
                Rs = Hs.T - numpy.dot(Ht.T, solve_chol(post.RC, Kts))  # (b x ns)
                Vb = linalg.solve(post.RHCH.T, Rs)  # (b x ns)
                covb = (Vb * Vb).sum(axis=0).reshape(-1, 1)
                sigma2 = covb + covf
            else:
                sigma2 = covf
            sigma2 = sigma2.flatten()
            return mu, sigma2
        else:
            return mu

    def _downdate(self, Ktt, Yt, i, Ht=None, computeSigma2=True):
        if self.post is None:
            raise RuntimeError('you should call fit or autoFit before')

        n = Ktt.shape[0]
        T = numpy.r_[numpy.arange(i), numpy.arange(i + 1, n)]
        Yt1 = Yt[T]

        # Covariance
        Kti = Ktt[T, i]
        Kii = Ktt[i, i]

        # Cholsky downdates (cf Osborne2010 p216)
        RC = self.post.RC
        RC11 = RC[:i, :i]
        RC13 = RC[:i, i + 1:]
        S23 = RC[i, i + 1:].copy()
        S33 = RC[i + 1:, i + 1:].copy()
        cholupdate(S33, S23)  # inplace
        RC33 = S33
        RC1 = numpy.r_[numpy.c_[RC11, RC13], numpy.c_[numpy.zeros(RC13.T.shape), RC33]]

        if Ht is not None:
            Ht1 = Ht[T, :]
            Hi = Ht[i, :]

            RHCH1 = cholpsd(numpy.dot(Ht1.T, solve_chol(RC1, Ht1)))

            # System resolution(cf RasmussenWilliams2006 Ch2 p28 Eq2.42)
            Ri = Hi - numpy.dot(Ht1.T, solve_chol(RC1, Kti))
            bet = solve_chol(RHCH1, numpy.dot(Ht1.T, solve_chol(RC1, Yt1)))
            invCY = solve_chol(RC1, (Yt1 - numpy.dot(Ht1, bet)))
            mu = numpy.dot(Hi.T, bet) + numpy.dot(Kti.T, invCY)
        else:
            invCY = solve_chol(RC1, Yt1)
            mu = numpy.dot(Kti.T, invCY)
            bet = None

        # sigma2
        if computeSigma2:
            Vf = linalg.solve(RC1.T, Kti)
            covf = Kii - (Vf * Vf).sum(axis=0).reshape(-1, 1)

            if Ht is not None:
                Vb = linalg.solve(RHCH1.T, Ri)
                covb = (Vb * Vb).sum(axis=0).reshape(-1, 1)
                sigma2 = covb + covf
            else:
                sigma2 = covf
            return mu, sigma2
        else:
            return mu

    def _loolik(self, Ktt, Yt, Ht=None):
        nll = 0

        for i in xrange(Ktt.shape[0]):
            mui, s2i = self._downdate(Ktt, Yt, i, Ht)
            nll = nll + .5 * \
                numpy.log(s2i) + (Yt[i] - mui)**2 / \
                (2 * s2i) + .5 * numpy.log(2 * numpy.pi)

        return nll
