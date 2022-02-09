'''Tools to analyse data with proper motions under the assumption that they are drawn
from an external rotating disc galaxy.'''
from ._AngleConversions import *
from ._GeometricFitting import findGradientsParametersFromData


def filterUsingProperMotions(data, a0=a0vdM, d0=d0vdM,
                             adjustForCorrelations=True,
                             defaultParallaxCut=5.,  # parallax/parallax_error < defaultParallaxCut
                             defaultPMChiSq=9.21,  # chi square for pm.: 9.21 -> 99% CI
                             defaultMagnitudeCut=20.,  # Magnitude cut for final sample
                             defaultMagnitudeCutInitialFilter=19.,  # Magnitude cut ini. sample
                             defaultRadiusCutInitialFilter=np.sin(
                                 np.deg2rad(3))  # Radius cut for initial sample
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

    # pmxLMC0, pmyLMC0, findMedianRobustCovariance(mux[maskpickpmLMC], muy[maskpickpmLMC])
    pmxLMC0 = np.median()
    pmyLMC0 = np.median(muy[maskpickpmLMC])

    # find Robust scatter estimate (RSE)
    # RSE is 0.39 * [(90th percentile) - (10th percentile)]
    # For a Gaussian it is the same the standard deviation (but it is more robust to outliers)

    percentilesx = np.percentile(data[maskpickpmLMC]['mux'], [10, 90])
    percentilesy = np.percentile(data[maskpickpmLMC]['muy'], [10, 90])
    RSEx = 0.390152*(percentilesx[1]-percentilesx[0])
    RSEy = 0.390152*(percentilesy[1]-percentilesy[0])
    maskPMRSE = ((np.absolute(data[maskpickpmLMC]['mux']-pmxLMC0) < 4*RSEx) &
                 (np.absolute(data[maskpickpmLMC]['muy']-pmyLMC0) < 4*RSEy))
    pmxcov = (data[maskpickpmLMC]['mux'])[maskPMRSE]
    pmycov = (data[maskpickpmLMC]['muy'])[maskPMRSE]
    pmcov = np.row_stack((pmxcov, pmycov))
    covar = np.cov(pmcov)
    icovar = np.linalg.inv(covar)

    print('Covarience matrix for proper motions (x & y)')
    print(covar)

    dpmx = np.array(data['mux']-pmxLMC0)
    dpmy = np.array(data['muy']-pmyLMC0)
    maskpmLMC = ((dpmx*dpmx*icovar[0][0] + 2*dpmx*dpmy*icovar[0]
                 [1] + dpmy*dpmy*icovar[1][1]) < defaultPMChiSq)

    pmxcov, pmycov, pmcov, dpmx, dpmy = [], [], [], [], []

    print(len(data[maskpmLMC]), ' stars fit initial proper motion cuts')
    maskposition = ((data['x']*data['x'] + data['y']*data['y'] <
                    defaultRadiusCutInitialFilter*defaultRadiusCutInitialFilter))
    maskmag = data['phot_g_mean_mag'] < defaultMagnitudeCutInitialFilter
    maskmyLMC = (maskparallaxLMC & maskpmLMC) & maskposition & maskmag
    median_parallax = np.median(np.array(data[maskmyLMC]['parallax']))

    print('median parallax of stars in LMC:', median_parallax, 'mas')

    pmra_decorr = (np.array(data.pmra) +
                   np.array(data.parallax_pmra_corr)*np.array(data.pmra_error) /
                   np.array(data.parallax_error)
                   * (median_parallax-np.array(data.parallax)))

    pmdec_decorr = (np.array(data.pmdec) +
                    np.array(data.parallax_pmdec_corr)*np.array(data.pmdec_error) /
                    np.array(data.parallax_error)
                    * (median_parallax-np.array(data.parallax)))

    data['pmra_decorr'] = pmra_decorr
    data['pmdec_decorr'] = pmdec_decorr

    # np.array(data.pmdec)
    _, _, LLmux, LLmuy = Spherical2Orthonormal(np.array(data.ra), np.array(data.dec),
                                               np.array(data.pmra_decorr),
                                               np.array(data.pmdec_decorr))

    data['mux_decorr'] = LLmux
    data['muy_decorr'] = LLmuy

    pmra_decorr, pmdec_decorr = [], []
    LLmux, LLmuy = [], [], [], []

    pmxLMC0_decorr = np.median(data[maskpickpmLMC]['mux_decorr'])
    pmyLMC0_decorr = np.median(data[maskpickpmLMC]['muy_decorr'])

    # find RSE
    percentilesx = np.percentile(data[maskpickpmLMC]['mux_decorr'], [10, 90])
    percentilesy = np.percentile(data[maskpickpmLMC]['muy_decorr'], [10, 90])
    RSEx = 0.390152*(percentilesx[1]-percentilesx[0])
    RSEy = 0.390152*(percentilesy[1]-percentilesy[0])
    maskPMRSE = ((np.absolute(data[maskpickpmLMC]['mux_decorr']-pmxLMC0) < 4*RSEx) &
                 (np.absolute(data[maskpickpmLMC]['muy_decorr']-pmyLMC0) < 4*RSEy))
    pmxcov = (data[maskpickpmLMC]['mux_decorr'])[maskPMRSE]
    pmycov = (data[maskpickpmLMC]['muy_decorr'])[maskPMRSE]
    # plt.hist(dpma,np.arange(-3,3,0.1),normed=True)
    # plt.plot(np.arange(-3,3,0.1),Gau(np.arange(-3,3,0.1),0,np.sqrt(0.214)),'--')
    pmcov = np.row_stack((pmxcov, pmycov))
    covar_decorr = np.cov(pmcov)
    icovar_decorr = np.linalg.inv(covar_decorr)

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
