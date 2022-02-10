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
    x, y, mux, muy = Spherical2Orthonormal_pandas(
        data, alpha0deg=a0, delta0deg=d0, convertUncertainties=False)
    # mask on parallax/parallax_error
    maskparallaxLMC = ((data.parallax)/(data.parallax_error) < defaultParallaxCut)
    # mask on default radius cut (restrict to LMC dominated region)
    maskposition = ((x**2 + y**2 < defaultRadiusCutInitialFilter*defaultRadiusCutInitialFilter))
    # mask anything too faint (again, reduces contamination because these have large uncertainties)
    maskbrightishLMC = data.phot_g_mean_mag < defaultMagnitudeCutInitialFilter

    maskpickpmLMC = maskparallaxLMC & maskbrightishLMC & maskposition

    pmxLMC0, pmyLMC0, covar = findMedianRobustCovariance(mux[maskpickpmLMC], muy[maskpickpmLMC])
    icovar = np.linalg.inv(covar)

    if verbose:
        print('Covarience matrix for proper motions (x & y)')
        print(covar)

    dpmx = np.array(mux-pmxLMC0)
    dpmy = np.array(muy-pmyLMC0)
    maskpmLMC = ((dpmx*dpmx*icovar[0][0] + 2*dpmx*dpmy*icovar[0]
                 [1] + dpmy*dpmy*icovar[1][1]) < defaultPMChiSq)

    dpmx, dpmy = [], [], [], [], []

    if verbose:
        print(len(data[maskpmLMC]), ' stars fit initial proper motion cuts')
    maskposition = (x**2+y**2 < defaultRadiusCutInitialFilter**2)
    maskmag = data['phot_g_mean_mag'] < defaultMagnitudeCutInitialFilter
    maskmyLMC = (maskparallaxLMC & maskpmLMC) & maskposition & maskmag
    median_parallax = np.median(np.array(data[maskmyLMC]['parallax']))
    if verbose:
        print('median parallax of stars after initial filter:', median_parallax, 'mas')

    # Return sample if they don't want to remove correlations

    pmra_decorr = (data.pmra.values +
                   data.parallax_pmra_corr.values * data.pmra_error.values
                   / data.parallax_error.values
                   * (median_parallax-data.parallax.values))

    pmdec_decorr = (data.pmdec.values +
                    data.parallax_pmdec_corr.values*data.pmdec_error.values /
                    data.parallax_error.values
                    * (median_parallax-data.parallax.values))

    _, _, LLmux, LLmuy = Spherical2Orthonormal(data.ra.values, data.dec.values,
                                               pmra_decorr,
                                               pmdec_decorr)

    data['mux'] = LLmux
    data['muy'] = LLmuy

    pmra_decorr, pmdec_decorr = [], []
    LLmux, LLmuy = [], []
    # TK replace with above function

    pmxLMC0_decorr, pmyLMC0_decorr, covar_decorr = findMedianRobustCovariance(data[maskpickpmLMC]['mux'],
                                                                              data[maskpickpmLMC]['muy'])
    
    icovar_decorr = np.linalg.inv(covar_decorr)
    if verbose: 
        print('Covarience matrix for proper motions (corrected x & y)')
        print(covar_decorr)

    dpmx_decorr = np.array(data['mux_decorr']-pmxLMC0_decorr)
    dpmy_decorr = np.array(data['muy_decorr']-pmyLMC0_decorr)
    maskpmLMC_decorr = ((dpmx_decorr*dpmx_decorr*icovar_decorr[0][0]
                         + 2*dpmx_decorr*dpmy_decorr*icovar_decorr[0][1]
                         + dpmy_decorr*dpmy_decorr*icovar_decorr[1][1]) < defaultPMChiSq)
    pmxcov, pmycov, pmcov, dpmx, dpmy = [], [], [], [], []
    maskMagnitudeFinal = (data['phot_g_mean_mag'] < defaultMagnitudeCut)
    maskLMCfinal = maskparallaxLMC & maskpmLMC_decorr & maskMagnitudeFinal

    print(len(data[maskLMCfinal]), 'stars in final sample')
