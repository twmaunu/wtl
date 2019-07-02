Files to run geodesic gradient descent for Robust Subspace Recovery and generate figures
in https://arxiv.org/abs/1706.03896.

-random pca: folder containing method for fast, randomized PCA from https://arxiv.org/abs/0809.2274

-calc_sdist: calculate distance between subspaces as the square root of the sum of the squared principal angles

-ggd: Geodesic gradient descent method from paper. See function for inputs. Can do 1/sqrt(k), line search, and ``shrinking’’ step size rules.

-ggd_test: script showing a test example for ggd on synthetic RSR task

-wtl_demo: function to generate various figures from the paper