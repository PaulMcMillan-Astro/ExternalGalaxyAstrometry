'''Tools to analyse data with proper motions under the assumption that they are drawn
from an external rotating disc galaxy.'''
from ._AngleConversions import *
from ._GeometricFitting import findGradientsParametersFromData


def findMedianRobustCovariance(mux, muy):
    '''Provides medians and a robust estimate of the covariance matrix of two input arrays'''
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
                             defaultParallaxCut=5.,  # parallax/parallax_error < defaultParallaxCut
                             defaultPMChiSq=9.21,  # chi square for pm.: 9.21 -> 99% CI
                             defaultMagnitudeCut=20.,  # Magnitude cut for final sample
                             defaultMagnitudeCutInitialFilter=19.,  # Magnitude cut ini. sample
                             defaultRadiusCutInitialFilter=np.sin(
                                 np.deg2rad(3)),  # Radius cut for initial sample
                             verbose=False
                             ):
    '''This function needs a lot of explaining'''

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
