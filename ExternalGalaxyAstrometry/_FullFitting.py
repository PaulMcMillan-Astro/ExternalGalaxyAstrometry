import scipy.optimize
import numpy as np
import _AngleConversions


def RotCurve_alpha(R, v0, r0, alpha):
    '''Rotation curve of a simple functional form: v0/(1+r0/R)^alpha)^(1/alpha)'''
    return v0/(1.+(r0/R)**alpha)**(1./alpha)


def ObjectiveFunctionRotCurve(params, x, y, mux, muy, mux_error, muy_error, mux_muy_corr):
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

    mux0, muy0, i, th, v0, R0, alpha_RC = params
    if R0 < 0.:
        return np.inf

    Rphi_out = _AngleConversions.Orthonormal2Internal(x, y, mux, muy,
                                                      mux_error, muy_error, mux_muy_corr,
                                                      mux0, muy0,
                                                      (_AngleConversions.vlos0vdM
                                                       / _AngleConversions.maskpc2kms
                                                       / _AngleConversions.D0LMC_Pietrzynski),
                                                      i, th)
    R, phi, vR, vphi, vR_error, vphi_error, vR_vphi_corr = Rphi_out

    # Model of average motions in the disc
    Vphi_mod = RotCurve_alpha(R, v0, R0, alpha_RC)
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


def fitRotationCurveModel(data, paramdict):
    '''Function which fits data to model of a rotating disc

    Parameters
    ----------
    data: pandas DataFrame
        This must contain the columns, ...
    paramdict:
        Dictionary containing the parameters decribed above (each either None or fixed value).

    Returns
    -------
    paramdict_out:
        Dictionary containing derived parameters

    '''

    keylist = ("a0", "d0", "mu_x0", "mu_y0", "mu_z0",
               "i", "Omega", "v0", "R0", "alpha_RC")

    # Read in dictionary and check what needs fitting, first question: a0 and d0?
    if all(k in paramdict for k in keylist):
        print("All parameters in dictionary. Continuing...")
    else:
        raise TypeError('Dictionary not complete. Requires all of: ' + keylist)
    #

    if (paramdict["a0"] is not None) and (paramdict["d0"] is not None):
        if (paramdict["mu_z0"] is not None):
            fitlist = ["mu_x0", "mu_y0", "i", "Omega", "v0", "R0", "alpha_RC"]
            params_fit = [paramdict(k) for k in fitlist]
            res = scipy.optimize.minimize(ObjectiveFunctionRotCurve, params_fit,
                                          args=(fitlist, data['x'].values, data['y'].values,
                                                data['muxBinned'].values, data['muyBinned'].values,
                                                data['muxUncertaintyBinned'].values,
                                                data['muyUncertaintyBinned'].values,
                                                data['muxmuyUncertaintyCorrBinned'].values))
        else:
            raise TypeError('Functionality not included (yet)')
    else:
        raise TypeError('Functionality not included (yet)')
