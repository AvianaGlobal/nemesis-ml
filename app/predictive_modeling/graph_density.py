# This file is a duplication of a cpp file (C++) in the Nemesis files
# The name is the same for both

# Calculating r combination of n elements.
# Note that This is the general wllknown formula and indeed is without repetition
# There might be faster way to calculate it. If the faster method is found this comment will be deleted.

from math import factorial as fact


def Comb(n, r):
    return (fact(n) / (fact(r) * fact(n - r)))
