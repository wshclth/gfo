# GFO (Generalized Feature Optimizer)
GFO is a lightweight (~180 lines) machine learning algorithm
that attemps to find a solution to Ax-b=0 where A is any
nxm matrix, and b is the wanted result vector. The solution x
gives weights to the features defined in A to obtain b as closely
as possible.

The A matrix is defined as so
| f_<sub>1</sub>(n) | f_<sub>2</sub>(n) | f_<sub>3</sub>(n) | ... | f_<sub>n</sub>(n) |
|:-----------------:|:-----------------:|:-----------------:|:---:|:-----------------:|
| f_<sub>1</sub>(0) | f_<sub>2</sub>(0) | f_<sub>3</sub>(0) |     |         ⋮         |
| f_<sub>1</sub>(1) | f_<sub>2</sub>(1) | f_<sub>3</sub>(1) |     |         ⋮         |
| f_<sub>1</sub>(2) | f_<sub>2</sub>(2) | f_<sub>3</sub>(2) |     |         ⋮         |
| f_<sub>1</sub>(3) | f_<sub>2</sub>(3) | f_<sub>3</sub>(3) |     |         ⋮         |
|       ⋮           |        ⋮          |         ⋮         |  ⋱  |         ⋮         |
| f_<sub>1</sub>(n) | f_<sub>2</sub>(n) | f_<sub>3</sub>(n) |     | f_<sub>3</sub>(n) |
