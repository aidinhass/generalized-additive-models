gam.py written by Ben Autrey

WARNING: All data in the domain X must be scaled between 0 and 1. This means that both training and test datasets must be scaled between 0 and 1 using the min and max across ALL data, e.g. (x - x_min)/x_max.

This is a Python implementation of an Additive Model found in chapter #3 of Simon Wood's Generalized Additive Models: An Introduction with R.

This is not truely an implementation of GAMs since I have not yet implemented all members of the exponential family of models, i.e. it is not "Generalized". This is a work in progress.

This implementation requires numpy, sklearn, and matplotlib.

To get the examples to run, do this:

1) Change directory to whereever you put gam.py.
2) Open a Python terminal.
3a) Copy and paste this into the terminal:

from gam import AdditiveModel
am = AdditiveModel()
am.generate_example_3()

OR

3b) Copy and paste this into the terminal:

from gam import AdditiveModel
am = AdditiveModel()
am.generate_example_2()

OR

3c) Copy and paste this into the terminal:

from gam import AdditiveModel
am = AdditiveModel()
am.generate_example_1()