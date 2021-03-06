#!/usr/bin/env python
from matplotlib import pyplot as plt

import pandas as pd
import numpy as np
import sys

class GFO:
    """Generalized Feature Optimizer. GFO is a lightweight dataset optimizer
    that is targeted towards fitting timeseries features to a derivative of
    the original time series.
    """

    def __init__(self, A, b, lr=1e-2, pre=None):
        """
        Given an A matrix of features with a wanted b set like so.


            |-------------------------------------|
            | f1(x) | f2(x) | f3(x) | ... | fs(x) |
            |=====================================|
            | f1(0) | f2(0) | f3(0) | ... | fn(0) |
            |-------|-------|-------|-----|-------|
        A = |   .   |   .   |  .    |  .  |   .   |
            |   .   |   .   |   .   |  .  |   .   |
            |   .   |   .   |    .  |  .  |   .   |
            |-------|-------|-------|-----|-------|
            | f1(m) | f2(m) | . . . . . . . fs(m) |
            |-------|-------|-------|-----|-------|

            where fs(m) is observation s of feature m
            
            |------|
            | r(0) |
            |------|
            | r(1) |
            |------|
            | .    |
        b = |  .   |
            |   .  |
            |------|
            | r(m) |
            |------|
            
            where r(m) is an observation to find a fit for.

        initalizes the learner to begin learning based on the given parameters.

        Neither A or b can have a NaN or complex numbers. Convergence is undefined
        in general complex numbers exist although. Convergence is always defined
        otherwise.
        """

        self.A = A
        self.b = b
        self.lr = lr
        self.pre = pre

    def __compute_residule__(self, A, x, b):
        """Computes the residual of the given vector using euclidian distance
        """
        r = A*x - b
        return np.sqrt(r.T*r).item()

    def __minimize_plane__(self, A, b, p, l):
        """Minimizes x by magnitude given the direction of the vector.
        Returns the optimized vector such that min(residule(A*x - l)) for
        the current cross section. Plane minimization does _not_ find
        directions, it assumes the direction of x is constant and the minimum
        exists by magnitude transformations

        lr is the learning rate of this function. The lower the learning rate
        the less precise the answer. The higher the learning rate the more
        precise the answer but the time needed to get to the answer increases
        exponentially.
        """
        
        # First optimize the magnitude. x*1.0 = x therefore 1.0 is the starting
        # weight
        magnitude = 0.0

        # mx is the minimum vector we are trying to find
        bp = p - b
        mx = b + (magnitude * bp)

        # The starting residual and an array to keep track of how the residual
        # changes via epoch.
        residual = self.__compute_residule__(A, mx, l)
        rh = [residual]

        while True:
            # Increase and decrease the magnitude via the learning rate.
            magnitudes = [magnitude, magnitude + self.lr, magnitude - self.lr]
            
            # Compute the residual for the new magnitudes
            residuals_ = [residual,
                          self.__compute_residule__(A, b + (magnitudes[1] * bp), l),
                          self.__compute_residule__(A, b + (magnitudes[2] * bp), l)]

            # Find the minimum residual
            mresidx = residuals_.index(min(residuals_))

            # If moving both left and right produces a worse residual then
            # the minimum has been found.
            if mresidx == 0:
                break
            else:
                magnitude = magnitudes[mresidx]
                rh.append(residuals_[mresidx])
                residual = rh[-1]
                mx = b + magnitudes[mresidx] * bp
        return mx, rh

    def learn(self, lrs=1):
        """Minimizes Ax-b=0 by finding a transformation vector x based on the
        principal components of the feature set A.
        """

        # Compute the COV of A
        cov = np.cov(self.A.T, bias=True)

        # Compute the eig values and vectors of the cov matrix
        w, vt = np.linalg.eig(cov)
        vt = np.matrix(vt)

        rhg = []
        steps = []

        if self.pre is None:
            x = np.zeros(vt[:, 0].shape)
        else:
            x = self.pre
        # Keeps track of the best full iteration
        last_best = None
        last_best_x = None
        global_epoch = 0
        local_epoch = 0
        for _ in range(len(w)):
            while True:
                Q, R = np.linalg.qr(x.T*vt)
                x = (Q*x.T).T
                for eigidx in range(0, x.shape[0]):
                    # Minimize plane defined by the direction of the given eigenvector
                    x, rh = self.__minimize_plane__(self.A, x, vt[:,eigidx], self.b)
                    steps.append(x)
                    local_epoch += 1
                    sys.stdout.flush()
                    print('global_epoch=%7d local_epoch=%7d residual=%32.16f' % (global_epoch, local_epoch, rh[-1]), end='\r')
                    sys.stdout.flush()
                local_epoch = 0
                global_epoch += 1
                rhg += rh
                if last_best is None:
                    last_best = rhg[-1]
                    last_best_x = steps[-1]
                else:
                    if last_best <= rhg[-1]:
                        break
                    else:
                        last_best = rhg[-1]
                        last_best_x = steps[-1]
                # print('residual: ', rhg[-1], end='\r')
                # Q, R = np.linalg.qr(x.T*vt)
                # x = (Q*x.T).T
            print('global_epoch=%7d local_epoch=%7d residual=%32.16f' % (global_epoch, local_epoch, rhg[-1]))
            sys.stdout.flush()
            self.lr /= 2.0
        return last_best_x, rhg, steps
