import numpy as np
import pandas as pd
from scipy.optimize import minimize
from ._AngleConversions import Spherical2Orthonormal

pd.set_option('mode.chained_assignment', None)


def minus_logLikelihood(mu_sig, x, xCov):
    """-logLikelihood, minimized to estimate true mean and dispersion (2D)

    Used by binDataOnSky. The likelihood is Gaussian with quoted mean and covarience
    matrix (described with 5 parameters).

    Parameters
    ----------
    mu_sig: array (5 values)
        means, dispersions and correlation coefficient (in that order)
    x: numpy array (Nx2)
        Input values
    xCov: numpy array (Nx2x2)
        Input uncertainties (in covariance matrix)

    Returns
    -------
    minusLogLike: float
        - log(likelihood)
    """

    if mu_sig[2] < 0.0:
        return 1e20
    if mu_sig[3] < 0.0:
        return 1e20
    if np.absolute(mu_sig[4]) >= 1.0:
        return 1e20

    mu = np.array(mu_sig[0:2])
    covInt = np.zeros([2, 2])
    covInt[0, 0] = mu_sig[2] ** 2
    covInt[1, 1] = mu_sig[3] ** 2
    covInt[0, 1] = covInt[1, 0] = mu_sig[2] * mu_sig[3] * mu_sig[4]

    covFull = xCov + covInt

    determinantFull = (
        covFull[:, 0, 0] * covFull[:, 1, 1] - covFull[:, 0, 1] * covFull[:, 0, 1]
    )
    invcovFull = np.zeros([xCov.shape[0], 2, 2])
    invcovFull[:, 0, 0] = covFull[:, 1, 1]
    invcovFull[:, 1, 1] = covFull[:, 0, 0]
    invcovFull[:, 0, 1] = -covFull[:, 0, 1]
    invcovFull[:, 1, 0] = -covFull[:, 1, 0]
    invcovFull = invcovFull / determinantFull.reshape(len(determinantFull), 1, 1)
    xPrime = x - mu
    exponent = np.einsum(
        "li,li->l", xPrime, np.einsum("lij,lj->li", invcovFull, xPrime)
    )
    # print(np.sum(np.log(determinantFull)), np.sum(exponent))
    return -(-0.5 * np.sum(np.log(determinantFull) + exponent))


def estimate_mean_dispersion_2D(x, xCov):
    """Estimate mean and intrinsic dispersion assuming 2D Gaussian

    Used by binDataOnSky to determine values for a single on-sky pixel.

    Parameters
    ----------
    x: numpy array (Nx2)
        Input values
    xCov: numpy array (Nx2x2)
        Input uncertainties (in covariance matrix)

    Returns
    -------
    estimate: array (5 values)
        From scipy.optimize.minimize - solution array x from OptimizeResult
    """
    guessCov = np.cov(x.T) - np.mean(xCov, axis=0)

    if guessCov[0, 0] <= 0.0:
        guessCov[0, 0] = 0.01
    if guessCov[1, 1] <= 0.0:
        guessCov[1, 1] = 0.01
    if guessCov[0, 1] ** 2 >= guessCov[1, 1] * guessCov[0, 0]:
        guessCov[0, 1] = guessCov[1, 0] = 0.0
    tmp = np.mean(x, axis=0)
    guessParameters = [
        tmp[0],
        tmp[1],  # Guess at weighted mean
        np.sqrt(guessCov[0, 0]),  # Guess at intrinsic dispersion
        np.sqrt(guessCov[1, 1]),  # Guess at intrinsic dispersion
        guessCov[0, 1] / np.sqrt(guessCov[0, 0] * guessCov[1, 1]),
    ]  # Guess correlation coeff
    #print(guessParameters)
    res = minimize(
        minus_logLikelihood,  # Function to minimize
        guessParameters,  # initial guess
        args=(x, xCov),
    )
    return res.x  # mu, sigmaInt


def binDataOnSky(data, a0, d0, angmax=8, statisticBins=40, verbose=True):
    """Compute the binned proper motion (with estimated uncertainty matrix) and dispersion

    Parameters
    ----------
    data: pandas DataFrame
        Must contain columns 'x','y','mux','muy','mux_error','muy_error','mux_muy_corr'
    angmax: float (default 8.)
        Data is binned between -angmax and angmax in x & y, where this value is in degrees
    statisticBins: int (default 40)
        Number of bins in x and y
    verbose: bool (default True)
        Give (somewhat) verbose output

    Returns
    -------
    data_out: pandas DataFrame
        Contains columns 'x', 'y', 'muxBinned', 'muyBinned',
        'muxDispersionBinned', 'muyDispersionBinned', 'muxmuyDispersionCorrBinned',
        'muxUncertaintyBinned', 'muyUncertaintyBinned':, 'muxmuyUncertaintyCorrBinned',
        'countBinned'

    """

    # Convert to orthonormal basis
    (
        data["x"],
        data["y"],
        data["mux"],
        data["muy"],
        data["mux_error"],
        data["muy_error"],
        data["mux_muy_corr"],
    ) = Spherical2Orthonormal(
        data["ra"],
        data["dec"],
        data["pmra"],
        data["pmdec"],
        data["pmra_error"],
        data["pmdec_error"],
        data["pmra_pmdec_corr"],
        a0,
        d0,
    )

    # x index
    xint = np.array(
        (np.floor((np.rad2deg(data.x) + angmax) * 0.5 * statisticBins / angmax))
    ).astype(int)

    # y index
    yint = np.array(
        (np.floor((np.rad2deg(data.y) + angmax) * 0.5 * statisticBins / angmax))
    ).astype(int)

    muxBinned = np.zeros([statisticBins, statisticBins])
    muyBinned = np.zeros([statisticBins, statisticBins])

    muxDispersionBinned = np.zeros([statisticBins, statisticBins])
    muyDispersionBinned = np.zeros([statisticBins, statisticBins])
    muxmuyDispersionCorrBinned = np.zeros([statisticBins, statisticBins])

    muxUncertaintyBinned = np.zeros([statisticBins, statisticBins])
    muyUncertaintyBinned = np.zeros([statisticBins, statisticBins])
    muxmuyUncertaintyCorrBinned = np.zeros([statisticBins, statisticBins])
    countBinned = np.zeros([statisticBins, statisticBins])

    xMedianBins = np.linspace(-angmax, angmax, statisticBins + 1)
    yMedianBins = np.linspace(-angmax, angmax, statisticBins + 1)

    xcentres = np.deg2rad(xMedianBins[0:-1] + 0.5 * (xMedianBins[1] - xMedianBins[0]))
    ycentres = np.deg2rad(yMedianBins[0:-1] + 0.5 * (yMedianBins[1] - yMedianBins[0]))

    xcentresfull, ycentresfull = np.meshgrid(xcentres, ycentres, indexing="ij")

    for i in range(statisticBins):
        for j in range(statisticBins):
            mask = (xint == i) & (yint == j)
            countBinned[i, j] = np.sum(mask)
            if countBinned[i, j] > 5:
                PM = np.vstack([data.mux.values[mask], data.muy.values[mask]]).T
                Cov = np.zeros([int(countBinned[i, j]), 2, 2])
                Cov[:, 0, 0] = data.mux_error[mask] ** 2
                Cov[:, 0, 1] = (
                    data.mux_error[mask]
                    * data.muy_error[mask]
                    * data.mux_muy_corr[mask]
                )
                Cov[:, 1, 0] = Cov[:, 0, 1]
                Cov[:, 1, 1] = data.muy_error[mask] ** 2
                meanDispersion = estimate_mean_dispersion_2D(PM, Cov)
                muxBinned[i, j] = meanDispersion[0]
                muyBinned[i, j] = meanDispersion[1]
                muxDispersionBinned[i, j] = np.absolute(meanDispersion[2])
                muyDispersionBinned[i, j] = np.absolute(meanDispersion[3])
                muxmuyDispersionCorrBinned[i, j] = meanDispersion[4]

                covUncertainty = np.zeros([2, 2])
                covUncertainty[0, 0] = np.sum(Cov[:, 0, 0] + meanDispersion[2] ** 2)
                covUncertainty[1, 1] = np.sum(Cov[:, 1, 1] + meanDispersion[3] ** 2)
                covUncertainty[0, 1] = np.sum(
                    Cov[:, 1, 0]
                    + meanDispersion[2] * meanDispersion[3] * meanDispersion[4]
                )
                covUncertainty[1, 0] = covUncertainty[0, 1]
                covUncertainty /= countBinned[i, j] ** 2
                muxUncertaintyBinned[i, j] = np.sqrt(covUncertainty[0, 0])
                muyUncertaintyBinned[i, j] = np.sqrt(covUncertainty[1, 1])
                muxmuyUncertaintyCorrBinned[i, j] = covUncertainty[1, 0] / (
                    muxUncertaintyBinned[i, j] * muyUncertaintyBinned[i, j]
                )
            else:
                muxBinned[i, j], muxDispersionBinned[i, j] = np.nan, np.nan
                muyBinned[i, j], muyDispersionBinned[i, j] = np.nan, np.nan
            if verbose:
                print(str(i) + "," + str(j) + "-", end="")
        if verbose:
            print(".", end="")
    if verbose:
        print("")

    fullBinnedData = {
        "x": xcentresfull.flatten(),
        "y": ycentresfull.flatten(),
        "muxBinned": muxBinned.flatten(),
        "muyBinned": muyBinned.flatten(),
        "muxDispersionBinned": muxDispersionBinned.flatten(),
        "muyDispersionBinned": muyDispersionBinned.flatten(),
        "muxmuyDispersionCorrBinned": muxmuyDispersionCorrBinned.flatten(),
        "muxUncertaintyBinned": muxUncertaintyBinned.flatten(),
        "muyUncertaintyBinned": muyUncertaintyBinned.flatten(),
        "muxmuyUncertaintyCorrBinned": muxmuyUncertaintyCorrBinned.flatten(),
        "countBinned": countBinned.flatten(),
    }

    return pd.DataFrame(data=fullBinnedData)
