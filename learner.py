#!/usr/bin/env python
from matplotlib import pyplot as plt

import pandas as pd
import numpy as np

class GFO:
    """Generalized Feature Optimizer. GFO is a lightweight dataset optimizer
    that is targeted towards fitting timeseries features to a derivative of
    the original time series.
    """

    def __init__(self, A, b):
        """
        Given an A matrix of features with a wanted b set like so.


            |-------------------------------------|
            | f1(x) | f2(x) | f3(x) | ... | fn(x) |
            |=====================================|
            | f1(0) | f2(0) | f3(0) | ... | fn(0) |
            |-------|-------|-------|-----|-------|
        A = |   .   |   .   |  .    |  .  |   .   |
            |   .   |   .   |   .   |  .  |   .   |
            |   .   |   .   |    .  |  .  |   .   |
            |-------|-------|-------|-----|-------|
            | f1(n) | f2(n) | . . . . . . . fn(n) |
            |-------|-------|-------|-----|-------|

            where fn(n) is observation n of feature n
            
            |------|
            | r(0) |
            |------|
            | r(1) |
            |------|
            | .    |
        b = |  .   |
            |   .  |
            |------|
            | r(n) |
            |------|
            
            where r(n) is an observation to find a fit for.

        initalizes the learner to begin learning based on the given parameters.

        Neither A or b can have a NaN or complex numbers. Convergence is undefined
        in general complex numbers exist although. Convergence is always defined
        otherwise.
        """

        self.A = A
        self.b = b

    def __compute_residule__(self, A, x, b):
        """Computes the residual of the given vector using euclidian distance
        """
        r = A*x - b
        return np.sqrt(r.T*r).item()

    def __minimize_plane__(self, A, x, b, lr=1e-4):
        """Minimizes x by magnitude given the direction of the vector.
        Returns the optimized vector such that min(residule(A*x - b)) for
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
        magnitude = 1.0

        # mx is the minimum vector we are trying to find
        mx = magnitude * x

        # The starting residual and an array to keep track of how the residual
        # changes via epoch.
        residual = self.__compute_residule__(A, mx, b)
        rh = [residual]

        while True:
            # Increase and decrease the magnitude via the learning rate.
            magnitudes = [magnitude, magnitude + lr, magnitude - lr]

            # Compute the residual for the new magnitudes
            residuals_ = [residual,
                          self.__compute_residule__(A, magnitudes[1]*mx, b),
                          self.__compute_residule__(A, magnitudes[2]*mx, b)]
            # Find the minimum residual
            mresidx = residuals_.index(min(residuals_))

            # If moving both left and right produces a worse residual then
            # the minimum has been found.
            if mresidx == 0:
                mx = x * magnitude
                break
            else:
                magnitude = magnitudes[mresidx]
                rh.append(residuals_[mresidx])
                residual = rh[-1]

        return mx, rh

    def learn(self):
        """Minimizes Ax-b=0 by finding a transformation vector x based on the
        principal components of the feature set A.
        """

        # Compute the COV of A
        cov = np.cov(self.A.T, bias=True)

        # Compute the eig values and vectors of the cov matrix
        w, vt = np.linalg.eig(cov)
        vt = np.matrix(vt)

        x = np.zeros(vt[:,0].shape)
        rhg = []
        steps = []
        for eigidx in range(x.shape[0]):
            # Nudge the vector into the next dimension by adding the current
            # eigenvector. x + lambda moves the x vector into a new plane.
            # Moving through each plane until all planes are optimized.
            # Moving through planes does not guantee a smaller residual as
            # time moves on.
            x, rh = self.__minimize_plane__(self.A, vt[:,eigidx]+x, self.b)
            steps.append(x)
            rhg += rh
            print('optized plane %3d / %3d, global_epoch=' % (eigidx+1,
                   x.shape[0]), len(rhg))

        return x, rhg, steps
