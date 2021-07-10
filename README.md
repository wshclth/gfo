# Generalized Feature Optimizer
A lightweight (~140 lines) machine learning algorithm that finds weights to features of an underlying data set.
This learner is geared towards timeseries data but can be used on any dataset. Due to the nature of this learner
underfitting and overfitting are extreamly rare, and in most cases impossible. This learner attempts to find a
vector x such that the residual of the equation Ax-b=0 is minimized, where A is a set of features, and b is the
underlying dataset that the features act upon.

This algorithm given any feature set (A) and any wanted data (b) will find a solution. It is up to the user to
understand the relationship of the feature set given and the data being to fit to. This algorithm works best
when features are _causations_ for changes in the underlying b vector.

For example the boiling point of water changes because of pressure and tempurature. In this case, pressure and
and tempurature would be features (A) and the recorded boiling point would be the wanted (b) vector.

Two example are given here, the classic winequality optimization where features of the wine are fit to the
(highly subjective) views of the drinker. And a financial example where we ask ourselves the question,
given an EMA12 and an EMA26, what is a good combination of these features to be used together?

Github has problems rendering some HTML you can view fully interactive notebooks on nbviewer.

https://nbviewer.ipython.org/github/wshclth/gfo/blob/master/financial.ipynb
https://nbviewer.ipython.org/github/wshclth/gfo/blob/master/winequality.ipynb