import numpy as np

a0vdM = 78.76 
'''a0 from van der Marel & Kallivayalil (2014) average of H I values'''
d0vdM = -69.19 
'''d0 from van der Marel & Kallivayalil (2014) average of H I values'''





def Spherical2Orthonormal(alphadeg,deltadeg,
                          mualphastar=None,mudelta=None,
                          mualphastar_error=None,mudelta_error=None, mualpha_mudelta_corr=None,
                          alpha0deg=a0vdM,delta0deg=d0vdM) :
    """Converts angular position, proper motion, and proper motion uncertainty to convenient local form 
    
    Input positions given in degrees, output in radians. 
    Proper motion in mas/yr (though I suppose pm units are unimportant).
    Conversion is to orthonormal x,y coordinates following eqs. 2 from DR2 demo paper
    
    Uncertainties do not include any uncertainty in position.
    
    Default alpha0, delta0 values (centre) from van der Marel & Kallivayalil (2014) 
    
    Outputs: 
        
        x,y: angular positions
        mux,muy: proper motions in x,y directions (if given as input)
        mux_error,muy_error,mux_muy_corr: uncertainties & correlation coefficient in proper motions (if given as input)
    """
    dalpharad = np.deg2rad(alphadeg-alpha0deg)
    
    cosda = np.cos(dalpharad)
    sinda = np.sin(dalpharad)
    deltarad = np.deg2rad(deltadeg)
    cosd = np.cos(deltarad)
    sind = np.sin(deltarad)
    delta0rad = np.deg2rad(delta0deg)
    cosd0 = np.cos(delta0rad)
    sind0 = np.sin(delta0rad)

    x = cosd*sinda
    y = sind*cosd0 - cosd*sind0*cosda
    if mualphastar is None or mudelta is None :
        return x,y
    if mualphastar_error is None or mudelta_error is None or mualpha_mudelta_corr is None : 
        mux = cosda*mualphastar-sind*sinda*mudelta
        muy = sind0*sinda*mualphastar+(cosd*cosd0+sind*sind0*cosda)*mudelta
        return x,y,mux,muy

    CovarianceMatrix = np.zeros((len(mualphastar_error),2,2))
    CovarianceMatrix[:,0,0] = mualphastar_error**2
    CovarianceMatrix[:,1,1] = mudelta_error**2
    CovarianceMatrix[:,0,1] = mualpha_mudelta_corr*mualphastar_error*mudelta_error
    CovarianceMatrix[:,1,0] = CovarianceMatrix[:,0,1]
    
    # Not really a rotation matrix.
    RotationMatrix = np.zeros((len(mualphastar),2,2))
    RotationMatrix[:,0,0] = cosda
    RotationMatrix[:,0,1] = -sind*sinda
    RotationMatrix[:,1,0] = sind0*sinda
    RotationMatrix[:,1,1] = (cosd*cosd0+sind*sind0*cosda)
    mu_alphadelta = np.vstack((mualphastar,mudelta)).T
    
    mu_xy = np.einsum('lij,lj->li', RotationMatrix, mu_alphadelta)
    mux = mu_xy[:,0]
    muy = mu_xy[:,1]
    
    # There may be a quicker way of doing this
    # C' = R C R^T
    # Multiply by TRANSPOSE of R (hence ij,kj not ij,jk)
    CovarianceMatrix_xy = np.einsum('lij,lkj->lik', CovarianceMatrix, RotationMatrix)
    # Multiply by R
    CovarianceMatrix_xy = np.einsum('lij,ljk->lik',  RotationMatrix,CovarianceMatrix_xy)
    mux_error = np.sqrt(CovarianceMatrix_xy[:,0,0])
    muy_error = np.sqrt(CovarianceMatrix_xy[:,1,1])
    mux_muy_corr = CovarianceMatrix_xy[:,0,1]/(mux_error*muy_error)
    return x,y,mux,muy,mux_error,muy_error,mux_muy_corr


def Orthonormal2Spherical(x,y,
                          mux=None,muy=None,
                          mux_error=None,muy_error=None, mux_muy_corr=None,
                          alpha0deg=a0vdM,delta0deg=d0vdM) :
    """Converts angular position and proper motion from convenient local form 
    
    Input positions in radians, outputs in degrees. Proper motion in mas/yr (though I suppose pm units are unimportant).
    Conversion is to orthonormal x,y coordinates following eqs. 2 from DR2 demo paper
    
    Default alpha0, delta0 values (centre) from van der Marel & Kallivayalil (2014) 
    """
    
    
    lambda0 = np.deg2rad(alpha0deg)
    phi0 = np.deg2rad(delta0deg)
    rho = np.sqrt(x**2+y**2)
    sinC = rho
    cosC = np.sqrt(1-sinC**2)
    
    tanLambda = x*sinC/(rho*cosC*np.cos(phi0)-y*sinC*np.sin(phi0))
    sinPhi = cosC*np.sin(phi0) + y*sinC*np.cos(phi0)/rho
    alphadeg,deltadeg = np.rad2deg(lambda0+np.arctan(tanLambda)),np.rad2deg(np.arcsin(sinPhi))

    if mux is None or muy is None :
        return alphadeg,deltadeg

    dalpharad = np.deg2rad(alphadeg-alpha0deg)
    
    cosda = np.cos(dalpharad)
    sinda = np.sin(dalpharad)
    deltarad = np.deg2rad(deltadeg)
    cosd = np.cos(deltarad)
    sind = np.sin(deltarad)
    delta0rad = np.deg2rad(delta0deg)
    cosd0 = np.cos(delta0rad)
    sind0 = np.sin(delta0rad)

    # Not really a rotation matrix.
    RotationMatrix = np.zeros((len(mux),2,2))
    RotationMatrix[:,0,0] = cosda
    RotationMatrix[:,0,1] = -sind*sinda
    RotationMatrix[:,1,0] = sind0*sinda
    RotationMatrix[:,1,1] = (cosd*cosd0+sind*sind0*cosda)
    
    Determinant = RotationMatrix[:,0,0]*RotationMatrix[:,1,1]-RotationMatrix[:,0,1]*RotationMatrix[:,1,0]
    # If it was really a rotation matrix the inverse would just be the transpose.
    InverseRotationMatrix = np.zeros((len(mux),2,2))
    InverseRotationMatrix[:,0,0] = RotationMatrix[:,1,1]/Determinant
    InverseRotationMatrix[:,0,1] = -RotationMatrix[:,0,1]/Determinant
    InverseRotationMatrix[:,1,0] = -RotationMatrix[:,1,0]/Determinant
    InverseRotationMatrix[:,1,1] = RotationMatrix[:,0,0]/Determinant
    
    mu_xy = np.vstack((mux,muy)).T
    
    mu_alphadelta = np.einsum('lij,lj->li', InverseRotationMatrix, mu_xy)
    mualphastar = mu_alphadelta[:,0]
    mudelta = mu_alphadelta[:,1]
    if mux_error is None or muy_error is None or mux_muy_corr is None :
        return alphadeg,deltadeg,mualphastar,mudelta
    
    CovarianceMatrix = np.zeros((len(mux_error),2,2))
    CovarianceMatrix[:,0,0] = mux_error**2
    CovarianceMatrix[:,1,1] = muy_error**2
    CovarianceMatrix[:,0,1] = mux_muy_corr*mux_error*muy_error
    CovarianceMatrix[:,1,0] = CovarianceMatrix[:,0,1]
    
    # There may be a quicker way of doing this
    # C' = R C R^T
    # Multiply by TRANSPOSE of R (hence ij,kj not ij,jk)
    CovarianceMatrix_ad = np.einsum('lij,lkj->lik', CovarianceMatrix, InverseRotationMatrix)
    # Multiply by R
    CovarianceMatrix_ad = np.einsum('lij,ljk->lik',  InverseRotationMatrix,CovarianceMatrix_ad)
    mualphastar_error = np.sqrt(CovarianceMatrix_ad[:,0,0])
    mudelta_error = np.sqrt(CovarianceMatrix_ad[:,1,1])
    mualpha_mudelta_corr = CovarianceMatrix_ad[:,0,1]/(mualphastar_error*mudelta_error)
    
    return alphadeg,deltadeg,mualphastar,mudelta,mualphastar_error,mudelta_error, mualpha_mudelta_corr



def Orthonormal2Internal(x,y,mux,muy,
                         vx,vy,vz,ideg,Omegadeg,
                         mux_error=None,muy_error=None,mux_muy_corr=None) :
    
    '''Given bulk motion/structure parameters derive rotation velocities and uncertainties

    See appendix of DR2 demo paper for mathematical details 

    N.B. Error in eq. B.21 of DR2 paper, which is now corrected in this code

    Inputs: 
        x,y: angular positions
        mux,muy: proper motions in x,y directions (if given as input)
        vx,vy,vz: Bulk velocity of LMC (units: mas/yr). vz is line-of-sight so conversion is at assumed distance
        ideg,Omegadeg: inclination & orientation of disc (degrees)
        mux_error,muy_error,mux_muy_corr: uncertainties & correlation coefficient in proper motions (if given as input)


    Outputs: 
        R (radians), phi (radians), vR (mas/yr), vt (mas/yr),
        (if mux_error, muy_error, mux_mu_corr given as input, also:)
        vR_error (mas/yr), vt_error (mas/yr), 
        vR_vt_corr (unit-free)

    vR/vt are in the same units as the input proper motion components.
    '''

    i = np.deg2rad(ideg)
    Omega = np.deg2rad(Omegadeg)
    a = np.tan(i)*np.cos(Omega)
    b = -np.tan(i)*np.sin(Omega)
    
    lx,mx,nx = np.sin(Omega),-np.cos(i)*np.cos(Omega),np.sin(i)*np.cos(Omega)
    ly,my,ny = np.cos(Omega),np.cos(i)*np.sin(Omega),-np.sin(i)*np.sin(Omega)
    lz,mz,nz = 0.0,np.sin(i),np.cos(i)

    z = np.sqrt(1.-x*x-y*y)
    
    Xi = (lx*x+ly*y)/(z+a*x+b*y) 
    Eta = ( (mx-a*mz)*x + (my-b*mz)*y ) / (z + a*x + b*y)
    R = np.sqrt(Xi*Xi+Eta*Eta)
    
    XiDotEtaDot2MuxMuyMatrix = np.zeros([len(x),2,2])
    
    XiDotEtaDot2MuxMuyMatrix[:,0,0] = (lx - x*(lx*x + ly*y)   )*(a*x+b*y+z)
    XiDotEtaDot2MuxMuyMatrix[:,0,1] = (mx - x*(mx*x+my*y+mz*z))*(a*x+b*y+z)
    XiDotEtaDot2MuxMuyMatrix[:,1,0] = (ly - y*(lx*x + ly*y)   )*(a*x+b*y+z)
    XiDotEtaDot2MuxMuyMatrix[:,1,1] = (my - y*(mx*x+my*y+mz*z))*(a*x+b*y+z)

    Determinant = (XiDotEtaDot2MuxMuyMatrix[:,0,0]*XiDotEtaDot2MuxMuyMatrix[:,1,1] 
                   - XiDotEtaDot2MuxMuyMatrix[:,1,0]*XiDotEtaDot2MuxMuyMatrix[:,0,1] )
    
    MuxMuy2XiDotEtaDotMatrix = np.zeros([len(x),2,2])
    
    MuxMuy2XiDotEtaDotMatrix[:,0,0] = XiDotEtaDot2MuxMuyMatrix[:,1,1]/Determinant
    MuxMuy2XiDotEtaDotMatrix[:,0,1] = -XiDotEtaDot2MuxMuyMatrix[:,0,1]/Determinant
    MuxMuy2XiDotEtaDotMatrix[:,1,0] = -XiDotEtaDot2MuxMuyMatrix[:,1,0]/Determinant
    MuxMuy2XiDotEtaDotMatrix[:,1,1] = XiDotEtaDot2MuxMuyMatrix[:,0,0]/Determinant
    

    mu_xy = np.vstack([ mux + (-vx + x*(vx*x+vy*y+vz*z)) * (a*x+b*y+z),
                        muy + (-vy + y*(vx*x+vy*y+vz*z)) * (a*x+b*y+z) ] ).T
    
    
    XiDotEtaDot2vRvtMatrix = np.zeros([len(x),2,2])
    XiDotEtaDot2vRvtMatrix[:,0,0] = Xi/R
    XiDotEtaDot2vRvtMatrix[:,0,1] = Eta/R
    XiDotEtaDot2vRvtMatrix[:,1,0] = -Eta/R
    XiDotEtaDot2vRvtMatrix[:,1,1] = Xi/R

    MuxMuy2vRvtMatrix = np.einsum('lij,ljk->lik', XiDotEtaDot2vRvtMatrix, MuxMuy2XiDotEtaDotMatrix)
    vRvt = np.einsum('lij,lj->li', MuxMuy2vRvtMatrix, mu_xy)
    
    if mux_error is None or muy_error is None or mux_muy_corr :
        return R, np.arctan2(Eta,Xi), vRvt[:,0], vRvt[:,1]

    CovarianceMatrix = np.zeros((len(x),2,2))
    CovarianceMatrix[:,0,0] = mux_error**2
    CovarianceMatrix[:,1,1] = muy_error**2
    CovarianceMatrix[:,0,1] = mux_muy_corr*mux_error*muy_error
    CovarianceMatrix[:,1,0] = CovarianceMatrix[:,0,1]

    # C' = R C R^T
    # Multiply by TRANSPOSE of R (hence ij,kj not ij,jk)
    CovarianceMatrix_vRvt = np.einsum('lij,lkj->lik', CovarianceMatrix, MuxMuy2vRvtMatrix)
    # Multiply by R
    CovarianceMatrix_vRvt = np.einsum('lij,ljk->lik', MuxMuy2vRvtMatrix, CovarianceMatrix_vRvt)
    
    vR_error = np.sqrt(CovarianceMatrix_vRvt[:,0,0])
    vt_error = np.sqrt(CovarianceMatrix_vRvt[:,1,1])
    vR_vt_corr = CovarianceMatrix_vRvt[:,0,1]/(vR_error*vt_error)
    return R, np.arctan2(Eta,Xi), vRvt[:,0], vRvt[:,1], vR_error, vt_error, vR_vt_corr

