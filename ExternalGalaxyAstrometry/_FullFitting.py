from operator import ge
from pytest import param
import scipy.optimize
import numpy as np
import warnings
from ._AngleConversions import *


def RotCurve_alpha(R, v0, r0, alpha):
    """Rotation curve of a simple functional form: v0/(1+r0/R)^alpha)^(1/alpha)"""
    return v0 / (1.0 + (r0 / R) ** alpha) ** (1.0 / alpha)


def ObjectiveFunctionRotCurve_XY(
    params, fixedParams, varyList, x, y, mux, muy, mux_error, muy_error, mux_muy_corr,
    systematic_uncertainty=0.
):
    """Function to minimize to perform a least-squares-like estimate of parameters.

    Can be used with minimizing function.

    The parameters contained within params must be:

    mux0, # bulk motion in x direction (mas/yr)
    muy0, # bulk motion in y direction (mas/yr)
    i,    # inclination of the disc (degrees)
    Omega,# orientation of the disc (degrees)
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
    systematic_uncertainty: float
        Systematic uncertainty to be added to the proper motion uncertainties

    Returns
    -------
    chi_sq: float
        Chi-square like statistic
    """
    
    keylist = [
        "a0",
        "d0",
        "mu_x0",
        "mu_y0",
        "mu_z0",
        "i",
        "Omega",
        "v0",
        "R0",
        "alpha_RC",
    ]

    paramdict = dict(zip(varyList, params))
    paramdict.update(fixedParams)

    if not all(k in paramdict.keys() for k in keylist):
        raise ValueError(
            "Input parameters missing from both fixedParams and guessParams: "
            + [k not in paramdict.keys() for k in keylist]
        )

    # mux0, muy0, i, th, v0, R0, alpha_RC = params
    if paramdict["R0"] < 0.0:
        return 1e20

    Rphi_out = Orthonormal2Internal(
        x,
        y,
        mux,
        muy,
        paramdict["mu_x0"],
        paramdict["mu_y0"],
        paramdict["mu_z0"],
        paramdict["i"],
        paramdict["Omega"],
        np.sqrt(mux_error**2 + systematic_uncertainty**2),
        np.sqrt(muy_error**2 + systematic_uncertainty**2),
        mux_muy_corr
    )
    R, phi, vR, vphi, vR_error, vphi_error, vR_vphi_corr = Rphi_out

    # Model of average motions in the disc
    Vphi_mod = RotCurve_alpha(
        R, paramdict["v0"], paramdict["R0"], paramdict["alpha_RC"]
    )
    VR_mod = 0.0 * np.ones_like(R)

    dv = np.vstack([vR - VR_mod, vphi - Vphi_mod]).T
    Cov = np.zeros([len(R), 2, 2])
    Cov[:, 0, 0] = vR_error**2
    Cov[:, 0, 1] = vR_error * vphi_error * vR_vphi_corr
    Cov[:, 1, 0] = Cov[:, 0, 1]
    Cov[:, 1, 1] = vphi_error**2
    invCov = np.linalg.inv(Cov)
    # chi^2 = dv' C^(-1) dv
    chi_sq_terms = np.einsum("li,li->l", dv, np.einsum("lij,lj->li", invCov, dv))
    chi_sq = np.sum(chi_sq_terms[np.isfinite(chi_sq_terms)])
    return chi_sq


def ObjectiveFunctionRotCurve_AD(
    params,
    fixedParams,
    varyList,
    alphadeg,
    deltadeg,
    mualphastar,
    mudelta,
    mualphastar_error,
    mudelta_error,
    mualpha_mudelta_corr,
    systematic_uncertainty=0.
):
    """Function to minimize to perform a least-squares-like estimate of parameters.

    Can be used with minimizing function, with input data in ra and dec.

    The parameters contained the two parameter dictionaries must be:

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
    alphadeg,delatdeg: numpy arrays
        Positions on the sky in ra and dec
    mualphastar,mudec: numpy arrays
        Proper motions (mas/yr)
    mualphastar_error,mudec_error,mualphastar_mudelta_corr: numpy arrays
        Uncertainties and their correlations (mas/yr, mas/yr, dimensionless, respectively)
    systematic_uncertainty: float
        Systematic uncertainty to be added to the proper motion uncertainties

    Returns
    -------
    chi_sq: float
        Chi-square like statistic
    """
    if ("a0" in varyList) and ("d0" in varyList):
        paramdictTmp = dict(
            zip(varyList, params)
        )  # Easy way of making sure this is done right
        a0deg = paramdictTmp["a0"]
        d0deg = paramdictTmp["d0"]
    else:
        paramdictTmp = dict(zip(varyList, params))
        paramdictTmp.update(fixedParams)
        a0deg = paramdictTmp["a0"]
        d0deg = paramdictTmp["d0"]

    tmp = Spherical2Orthonormal(
        alphadeg,
        deltadeg,
        mualphastar,
        mudelta,
        mualphastar_error,
        mudelta_error,
        mualpha_mudelta_corr,
        a0deg,
        d0deg,
    )
    x, y, mux, muy, mux_error, muy_error, mux_muy_corr = tmp
    return ObjectiveFunctionRotCurve_XY(
        params,
        fixedParams,
        varyList,
        x,
        y,
        mux,
        muy,
        mux_error,
        muy_error,
        mux_muy_corr,
        systematic_uncertainty,
    )


def fitRotationCurveModel(data, fixedParams, guessParams, systematic_uncertainty=0.):
    """Function which fits data to model of a rotating disc

    Parameters of the disc are given in two dictionaries containing the fixed parameters
    and their values, and the variable parameters and an initial guess of their values.
    Between the two, all of the following must be defined

    "a0", "d0": The centre of the disc rotation in degrees (Note 1)
    "mu_x0", "mu_y0": bulk motion in the orthonormal x,y coordinates [mas/yr]
    "mu_z0": Line-of-sight velocity, positive for motion away [mas/yr] (Note 2)
    "i", "Omega": Disc inclination and orientation
    "v0", "R0", "alpha_RC": Parameters of disc rotation (Note 3)

    Note 1: If a0 and d0 are to be fit then the initial guess values must be the same as those
        used to find the input values x, y, etc. in the orthonormal coordinate system.
    Note 2: This is placed on the same scale as the proper motions, so depends on the distance to
        the centre of the LMC. For a line of sight velocity vlos km/s, and a galaxy at a
        distance Dgal, this would be vlos/Dgal/4.7403885, where the last value is a conversion
        factor between mas/yr and kms/kpc (stored as maskpc2kms in this package)

    Note 3: Rotation curve is of the form v0/(1+r0/R)^alpha)^(1/alpha).

    Parameters
    ----------
    data: pandas DataFrame
        This must contain the columns, 'x', 'y', 'muxBinned', 'muyBinned',
        'muxUncertaintyBinned', 'muyUncertaintyBinned', 'muxmuyUncertaintyCorrBinned'
    fixedParams: dictionary
        Contains the parameters with fixed values and their values
    guessParams: dictionary
        Contains parameters to be fit and initial estimated values
    systematic_uncertainty: float
        Systematic uncertainty to be added to the proper motion uncertainties

    Returns
    -------
    paramdict_out: Dictionary
        Dictionary containing derived parameters
    message:
        messsage returned from call of scipy.optimize.minimize

    """

    keylist = [
        "a0",
        "d0",
        "mu_x0",
        "mu_y0",
        "mu_z0",
        "i",
        "Omega",
        "v0",
        "R0",
        "alpha_RC",
    ]

    # Check that all parameters have input guesses
    allParamsInput = {}
    allParamsInput.update(fixedParams)
    allParamsInput.update(guessParams)

    # Check all parameters given
    if not all(k in allParamsInput.keys() for k in keylist):
        raise ValueError(
            "Input parameters missing from both fixedParams and guessParams: "
            + [k not in allParamsInput.keys() for k in keylist]
        )
    elif not all(k in keylist for k in allParamsInput.keys()):
        warnings.warn(
            "Input parameters not understood: "
            + [k not in keylist for k in allParamsInput.keys()]
        )
    else:
        print("All parameters given and understood. Continuing...")

    # Put parameter names into lists to avoid any issues with ordering of dicts
    varyList = [k for k in guessParams.keys()]
    guessList = [guessParams[k] for k in varyList]

    # Fixed centre: this simplifies the calculation as we do not need to recalculate x,y etc
    if ("a0" in fixedParams.keys()) and ("d0" in fixedParams.keys()):
        res = scipy.optimize.minimize(
            ObjectiveFunctionRotCurve_XY,
            guessList,
            args=(
                fixedParams,
                varyList,
                data["x"].values,
                data["y"].values,
                data["muxBinned"].values,
                data["muyBinned"].values,
                data["muxUncertaintyBinned"].values,
                data["muyUncertaintyBinned"].values,
                data["muxmuyUncertaintyCorrBinned"].values,
                systematic_uncertainty,
            ),
            options={"maxiter": 100000},
        )
    else:
        # Convert to ra and dec, so that each iteration can make conversion to x,y itself
        (
            alphadeg,
            deltadeg,
            mualphastar,
            mudelta,
            mualphastar_error,
            mudelta_error,
            mualpha_mudelta_corr,
        ) = Orthonormal2Spherical(
            data["x"].values,
            data["y"].values,
            data["muxBinned"].values,
            data["muyBinned"].values,
            data["muxUncertaintyBinned"].values,
            data["muyUncertaintyBinned"].values,
            data["muxmuyUncertaintyCorrBinned"].values,
            alpha0deg=guessParams["a0"],
            delta0deg=guessParams["d0"],
        )
        res = scipy.optimize.minimize(
            ObjectiveFunctionRotCurve_AD,
            guessList,
            args=(
                fixedParams,
                varyList,
                alphadeg,
                deltadeg,
                mualphastar,
                mudelta,
                mualphastar_error,
                mudelta_error,
                mualpha_mudelta_corr,
                systematic_uncertainty,
            ),
            options={"maxiter": 100000}
        )
    if not res.success:
        warnings.warn("minimization did not complete successfully")
    return dict(zip(varyList, res.x)), res.message
