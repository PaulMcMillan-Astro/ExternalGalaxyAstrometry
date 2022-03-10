'''Tools to analyse data with proper motions under the assumption that they are drawn
from an external rotating disc galaxy.'''
from ._AngleConversions import *
from ._GeometricFitting import findGradientsParametersFromData


def findMedianRobustCovariance(mux, muy):
    '''Provides medians and a robust estimate of the covariance matrix of two input arrays

    Covarience matrix is estimated by calculating the covarience of all values within 4 Robust
    Scatter Estimates (RSE) of the median in both mux and muy. The RSE is calculated as

    RSE = 0.390152 * (Q(0.9) - Q(0.1))

    for quantiles Q.

    Parameters
    ----------
    mux: array of floats
        values for calculation
    muy: array of floats
        values for calculation

    Returns
    -------
    pmx0: float
        median in mux
    pmy0: float
        median in muy
    covar: array of float
        Robust covarience estimate. Dimensions (2,2)

    '''
    pmx0 = np.median(mux)
    pmy0 = np.median(muy)

    percentilesx = np.percentile(mux, [10, 90])
    percentilesy = np.percentile(muy, [10, 90])
    RSEx = 0.390152*(percentilesx[1]-percentilesx[0])
    RSEy = 0.390152*(percentilesy[1]-percentilesy[0])
    maskPMRSE = ((np.absolute(mux-pmx0) < 4*RSEx) &
                 (np.absolute(muy-pmy0) < 4*RSEy))
    pmxcov = (mux)[maskPMRSE]
    pmycov = (muy)[maskPMRSE]
    pmcov = np.row_stack((pmxcov, pmycov))
    covar = np.cov(pmcov)
    return pmx0, pmy0, covar


def filterUsingProperMotions(data, a0=a0vdM, d0=d0vdM,
                             adjustForCorrelations=True,
                             defaultParallaxCut=5.,
                             defaultPMChiSq=9.21,
                             defaultMagnitudeCut=20.,
                             defaultMagnitudeCutInitialFilter=19.,
                             defaultRadiusCutInitialFilter=np.sin(np.deg2rad(3)),
                             verbose=False
                             ):
    '''Function which filters data using proper motions

    Parameters
    ----------
    data: pandas DataFrame
        DataFrame containing stars in the region on sky of the galaxy containing (at
        minimum) the columns described above.
    a0: float (default a0 for LMC from van der Marel TK)
        Assumed centre of the galaxy in RA.
    d0: float (default d0 for LMC from van der Marel TK)
        Assumed centre of the galaxy in declination.
    adjustForCorrelations: bool (default True)
        Adjust proper motions taking into account correlated uncertainties and
        parallax of the galaxy.
    defaultParallaxCut: float (default 5.0)
        remove stars with parallax/parallax_error < defaultParallaxCut as quality filter.
    defaultPMChiSq: float (default 9.21)
        Threshold for proper motion cut in terms of intrinsic dispersion (9.21 -> 99% CR).
    defaultMagnitudeCut: float (default 20.0)
        Magnitude cut for final sample
    defaultMagnitudeCutInitialFilter: float (default 19.0)
        Magnitude cut for initial sample used to define proper motion dispersion.
    defaultRadiusCutInitialFilter: float (default sin(3 deg))
        Radius cut for initial sample used to define proper motion dispersion.
    verbose: bool (default False)
        Give verbose output.

    Returns
    -------
    data: pandas DataFrame
        The input DataFrame with new columns 'x', 'y', 'mux', 'muy'

    '''

    x, y, mux, muy = Spherical2Orthonormal_pandas(data, alpha0deg=a0, delta0deg=d0,
                                                  convertUncertainties=False)

    # masks for initial sample to estimate proper motion covariance
    maskIniParallax = ((data.parallax)/(data.parallax_error) < defaultParallaxCut)
    maskIniPosition = (x**2 + y**2 < defaultRadiusCutInitialFilter**2)
    maskIniMagnitude = data.phot_g_mean_mag < defaultMagnitudeCutInitialFilter
    maskFinalMagnitude = (data.phot_g_mean_mag < defaultMagnitudeCut)
    maskpickpmLMC = maskIniParallax & maskIniMagnitude & maskIniPosition

    pmxLMC0, pmyLMC0, covar = findMedianRobustCovariance(mux[maskpickpmLMC], muy[maskpickpmLMC])
    icovar = np.linalg.inv(covar)

    if verbose:
        print('Covarience matrix for proper motions (x & y)')
        print(covar)

    dpmx = np.array(mux-pmxLMC0)
    dpmy = np.array(muy-pmyLMC0)
    maskpmLMC = ((dpmx*dpmx*icovar[0][0] + 2*dpmx*dpmy*icovar[0]
                 [1] + dpmy*dpmy*icovar[1][1]) < defaultPMChiSq)

    if verbose:
        print(len(data[maskpmLMC]), ' stars fit initial proper motion cuts')

    maskmyLMC = (maskIniParallax & maskpmLMC) & maskIniPosition & maskIniMagnitude
    median_parallax = np.median(np.array(data[maskmyLMC]['parallax']))
    if verbose:
        print('median parallax of stars after initial filter:', median_parallax, 'mas')

    # Return sample if they don't want to remove correlations
    if adjustForCorrelations is False:
        data['mux'] = mux
        data['muy'] = muy
        maskmyLMC = (maskIniParallax & maskpmLMC) & maskFinalMagnitude
        return data[maskmyLMC]

    pmra_decorr = (data.pmra.values +
                   data.parallax_pmra_corr.values * data.pmra_error.values
                   / data.parallax_error.values
                   * (median_parallax-data.parallax.values))

    pmdec_decorr = (data.pmdec.values +
                    data.parallax_pmdec_corr.values*data.pmdec_error.values /
                    data.parallax_error.values
                    * (median_parallax-data.parallax.values))

    _, _, mux, muy = Spherical2Orthonormal(data.ra.values, data.dec.values,
                                           pmra_decorr,
                                           pmdec_decorr)

    pmra_decorr, pmdec_decorr = [], []

    pmxLMC0, pmyLMC0, covar = findMedianRobustCovariance(mux[maskpickpmLMC],
                                                         muy[maskpickpmLMC])
    icovar = np.linalg.inv(covar)

    if verbose:
        print('Covarience matrix for proper motions (corrected x & y)')
        print(covar)

    dpmx = mux-pmxLMC0
    dpmy = muy-pmyLMC0
    maskpmLMC = ((dpmx*dpmx*icovar[0][0] + 2*dpmx*dpmy*icovar[0][1]
                  + dpmy*dpmy*icovar[1][1]) < defaultPMChiSq)

    maskLMCfinal = maskIniParallax & maskpmLMC & maskFinalMagnitude

    if verbose:
        print(len(data[maskLMCfinal]), 'stars in final sample')

    data['mux'] = mux
    data['muy'] = muy
    return data[maskLMCfinal]
