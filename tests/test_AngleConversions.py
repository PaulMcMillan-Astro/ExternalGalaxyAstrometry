import ExternalGalaxyAstrometry as EGA
import numpy as np


def test_Spherical2Orthonormal_with_Orthonormal2Spherical():
    """ It's my first test!"""
    tol = 1e-8
    assert (EGA.Spherical2Orthonormal(np.array([80]), np.array([80]), np.array([4]), np.array([4]), 
                                      np.array([0.1]), np.array([0.1]), np.array([0.5]), 
                                      80, 80))[0][0] < tol, \
        """Spherical2Orthonormal centring wrong"""
    return None
