## GFO (Generalized Feature Optimizer)
GFO is a lightweight (~180 lines) machine learning algorithm
that attemps to find a solution to Ax-b=0 where A is any
nxm matrix, and b is the wanted result vector. The solution x
gives weights to the features defined in A to obtain b as closely
as possible.

The A matrix is defined as so
| f_<sub>1</sub>(n) | f_<sub>2</sub>(n) | f_<sub>3</sub>(n) | ... | f_<sub>s</sub>(n) |
|:-----------------:|:-----------------:|:-----------------:|:---:|:-----------------:|
| f_<sub>1</sub>(0) | f_<sub>2</sub>(0) | f_<sub>3</sub>(0) |     |         ⋮         |
| f_<sub>1</sub>(1) | f_<sub>2</sub>(1) | f_<sub>3</sub>(1) |     |         ⋮         |
| f_<sub>1</sub>(2) | f_<sub>2</sub>(2) | f_<sub>3</sub>(2) |     |         ⋮         |
| f_<sub>1</sub>(3) | f_<sub>2</sub>(3) | f_<sub>3</sub>(3) |     |         ⋮         |
|       ⋮           |        ⋮          |         ⋮         |  ⋱  |         ⋮         |
| f_<sub>1</sub>(m) | f_<sub>2</sub>(m) | f_<sub>3</sub>(m) |     | f_<sub>s</sub>(m) |

where f_<sub>s</sub>(m) is the value of feature s at observation m.

The b matrix is defined as so
| c(n) |
|:----:|
| c(0) |
| c(1) |
|  ⋮   |
| c(m) |

where c(m) is the value of the wanted value at observation m.
GFO minimizes Ax-b=0 by using the principal components of A defined by the
eigenvectors of the Cov(A<sup>T</sup>) as search directions. Starting with
x = [0 0 0 ... 0]<sup>T</sup> (or a given start vector)
we minimize x in the direction of the first eigenvector. After,
minimize in the direction of the second eigenvector.
Keep minimizing and looping through each eigenvector until a minimim is found.
Because Cov(A<sup>T</sup>) is always positive definite, convergence is guanteed
to a minimum, but the minimum converged to through the directions defined by
eig(Cov(A<sup>T</sup>)) is not guanteed to be the global minimum.
