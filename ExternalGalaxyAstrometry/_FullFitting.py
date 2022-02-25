from operator import ge
from pytest import param
import scipy.optimize
import numpy as np
import _AngleConversions


def RotCurve_alpha(R, v0, r0, alpha):
    '''Rotation curve of a simple functional form: v0/(1+r0/R)^alpha)^(1/alpha)'''
    return v0/(1.+(r0/R)**alpha)**(1./alpha)


def ObjectiveFunctionRotCurve_XY(params, fixedParams, varyList,
                                 x, y, mux, muy, mux_error, muy_error, mux_muy_corr):
    '''Function to minimize to perform a least-squares-like estimate of parameters.

    Can be used with minimizing function.

    The parameters contained within params must be:

    mux0, # bulk motion in x direction (mas/yr)
    muy0, # bulk motion in y direction (mas/yr)
    i,    # inclination of the disc (degrees)
    th,   # orientation of the disc (degrees)
    v0,   # v0 for rotation curve (see above, mas/yr)
    R0,   # R0 for rotation curve (see above, radians)
    alpha_RC # alpha for rotation curve (dimensionless)

    Note that everything is scaled to the distance of the galaxy,
    so R0 is real distance/distance to LMC, and
    v0 is (velocity in kms)/(distance to LMC in kpc)/(conversion factor from kpc mas/yr to kms)

    Parameters
    ----------
    params: list-like
        Parameters of model, see description above
    x,y: numpy arrays
        Positions of the sky relative to a pre-defined centre (radians)
    mux,muy: numpy arrays
        Proper motions (mas/yr)
    mux_error,muy_error,mux_muy_corr: numpy arrays
        Uncertainties and their correlations (mas/yr, mas/yr, dimensionless, respectively)

    Returns
    -------
    chi_sq: float
        Chi-square like statistic
    '''
    keylist = ["a0", "d0", "mu_x0", "mu_y0", "mu_z0",
               "i", "Omega", "v0", "R0", "alpha_RC"]

    paramdict = dict(zip(varyList, params))
    paramdict.update(fixedParams)

    if not all(k in paramdict.keys() for k in keylist):
        raise ValueError("Input parameters missing from both fixedParams and guessParams: "
                         + [k not in paramdict.keys() for k in keylist])

    # mux0, muy0, i, th, v0, R0, alpha_RC = params
    if paramdict['R0'] < 0.:
        return np.inf

    Rphi_out = _AngleConversions.Orthonormal2Internal(x, y, mux, muy,
                                                      mux_error, muy_error, mux_muy_corr,
                                                      paramdict['mu_x0'], paramdict['mu_y0'],
                                                      paramdict['mu_z0'],
                                                      paramdict['i'], paramdict['th'])
    R, phi, vR, vphi, vR_error, vphi_error, vR_vphi_corr = Rphi_out

    # Model of average motions in the disc
    Vphi_mod = RotCurve_alpha(R, paramdict['v0'], paramdict['R0'], paramdict['alpha_RC'])
    VR_mod = 0.*np.ones_like(R)

    dv = np.vstack([vR-VR_mod, vphi-Vphi_mod]).T
    Cov = np.zeros([len(R), 2, 2])
    Cov[:, 0, 0] = vR_error**2
    Cov[:, 0, 1] = vR_error*vphi_error*vR_vphi_corr
    Cov[:, 1, 0] = Cov[:, 0, 1]
    Cov[:, 1, 1] = vphi_error**2
    invCov = np.linalg.inv(Cov)
    # chi^2 = dv' C^(-1) dv
    chi_sq_terms = np.einsum('li,li->l', dv, np.einsum('lij,lj->li', invCov, dv))
    chi_sq = np.sum(chi_sq_terms[np.isfinite(chi_sq_terms)])
    return chi_sq


def fitRotationCurveModel(data, fixedParams, guessParams):
    '''Function which fits data to model of a rotating disc

    Parameters
    ----------
    data: pandas DataFrame
        This must contain the columns, ...
    fixedParams:
        Dictionary containing the parameters with fixed values and their values
    guessParams:
        Dictionary containing parameters to be fit and initial estimated values

    Returns
    -------
    paramdict_out:
        Dictionary containing derived parameters (including fixed ones)

    '''

    keylist = ["a0", "d0", "mu_x0", "mu_y0", "mu_z0",
               "i", "Omega", "v0", "R0", "alpha_RC"]

    # Check that all parameters have input guesses
    allParamsInput = {}
    allParamsInput.update(fixedParams)
    allParamsInput.update(guessParams)

    # Check all parameters given
    if not all(k in allParamsInput.keys() for k in keylist):
        raise ValueError("Input parameters missing from both fixedParams and guessParams: "
                         + [k not in allParamsInput.keys() for k in keylist])
    elif not all(k in keylist for k in allParamsInput.keys()):
        raise ValueError("Input parameters not understood: "
                         + [k not in keylist for k in allParamsInput.keys()])
    else:
        print('All parameters given and understood. Continuing...')

    # Fixed centre: this simplifies the calculation as we do not need to recalculate x,y etc
    if ("a0" in fixedParams.keys()) and ("d0" in fixedParams.keys()):
        # Put parameter names into lists to avoid any issues with ordering of dicts
        varyList = [k for k in guessParams.keys()]
        guessList = [guessParams[k] for k in varyList]
        res = scipy.optimize.minimize(ObjectiveFunctionRotCurve_XY, guessList,
                                      args=(fixedParams, varyList,
                                            data['x'].values, data['y'].values,
                                            data['muxBinned'].values, data['muyBinned'].values,
                                            data['muxUncertaintyBinned'].values,
                                            data['muyUncertaintyBinned'].values,
                                            data['muxmuyUncertaintyCorrBinned'].values))
    else:
        raise TypeError('Functionality not included (yet)')
