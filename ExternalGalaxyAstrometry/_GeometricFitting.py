import numpy as np
import scipy.optimize

def EFGHgivenDirectVals(vx,dmxdx,dmxdy,vy,dmydx,dmydy,a,b) :
    '''Helper function for solving equations to find vz,i,Omega,omega.'''
    E = dmxdx+dmydy-(a*vx+b*vy)
    F = dmxdx-dmydy-(a*vx-b*vy)
    G = dmxdy+dmydx-(b*vx+a*vy)
    H = dmxdy-dmydx-(b*vx-a*vy)
    return E,F,G,H

def vzcOmomgivenEFGH(E,F,G,H) :
    '''Helper function for solving equations to find vz,i,Omega,omega.'''
    vz = -0.5*E
    c = - np.sign(H) * np.sqrt(np.absolute(( np.absolute(H) - np.sqrt(F*F+G*G) ) /
                                (np.absolute(H) + np.sqrt(F*F+G*G))))
    Om = 0.5*np.arctan2(-F*H,-G*H)
    om = -c*H/(1+c*c)
    return vz,c,Om,om

def igivenc(c) :
    '''Helper function for solving equations to find vz,i,Omega,omega.'''
    i = np.arccos(c)
    return i

def abgiveniOm(i, Om) :
    '''Helper function for solving equations to find vz,i,Omega,omega.'''
    a = np.tan(i)*np.cos(Om)
    b = -np.tan(i)*np.sin(Om)
    return a,b

def findGalaxyParametersFromGradients(vx,dmxdx,dmxdy,vy,dmydx,dmydy,
                                        iGuess=0.,OmGuess=0.) :
    '''Directly give vz,i,Omega,omega given gradients of proper motion field

    Input estimates of the inclination and Omega can be given
    '''
    a = 0
    b = 0
    a,b = abgiveniOm(np.deg2rad(iGuess),np.deg2rad(OmGuess)) # initial
    for i in range(150) : # iterative solution
        E,F,G,H = EFGHgivenDirectVals(vx,dmxdx,dmxdy,vy,dmydx,dmydy,a,b)
        vz,c,Om,om = vzcOmomgivenEFGH(E,F,G,H)
        #print(vz,c,Om,om)
        i = igivenc(c)
        #print(i)
        a, b = abgiveniOm(i, Om)
    return vx,vy,vz,np.rad2deg(i),np.rad2deg(Om),om

def ObjectFuncForGradientFinding(params, mu, x, y) :
    '''Objective function, for mu_model = param[0] + param[1]*x + param[2]*y
    
    Computes sum (mu-mu_model)^2 for all points (cutting non-finite ones)
    '''
    mu_mod = params[0] + params[1]*x + params[2]*y #plane(params,x, y)
    chi_2_i = np.power(mu-mu_mod,2)
    chi_2_i = chi_2_i.flatten() # This feels unnecessary
    chi_2_good = chi_2_i[np.isfinite(chi_2_i)]
    return np.sum(chi_2_good)



def findGradientsParametersFromData(data,muxname='mux_decorr',muyname='muy_decorr') :
    '''Given input proper motion data, estiamtes properties of a disc galaxy

    Input: data is a pandas dataframe with columns:
        'x','y' (angles on sky)
        muxname,muyname (name for proper motions in those directions)

    outputs: vx,vy,vz,ideg,Omdeg,om,dmxdx,dmxdy,dmydx,dmydy (floats)
    
    vx,vy,vz are bulk motions in x,y,z direction (z = line-of-sight) in units of proper motion
    ideg is inclination (in degrees)
    Omdeg is orientation of disc (in degrees)
    om is angular velocity (units: proper motion/radians)
    the dmA/dB is the gradient in proper motion mA with respect to direction B
    '''
    parameters0 = np.array([np.mean(data[muxname]),-1.3,-3.]) # Arbitrary-ish values, might be best to fix at some point
    resx = scipy.optimize.minimize(ObjectFuncForGradientFinding,parameters0,
                        args=(np.array(data[muxname]),np.array(data['x']), np.array(data['y'])),
                        method='Nelder-Mead',options={'maxiter' : 500})

    parameters0 = np.array([ np.mean(data[muyname]) , 4., -0.5]) # Arbitrary-ish values, might be best to fix at some point
    resy = scipy.optimize.minimize(ObjectFuncForGradientFinding,parameters0,
                        args=(np.array(data[muyname]),np.array(data['x']), np.array(data['y'])),
                        method='Nelder-Mead',options={'maxiter' : 500})
    vx = resx.x[0]
    dmxdx = resx.x[1]
    dmxdy = resx.x[2]
    vy = resy.x[0]
    dmydx = resy.x[1]
    dmydy = resy.x[2]
    vx,vy,vz,ideg,Omdeg,om = findGalaxyParametersFromGradients(vx,dmxdx,dmxdy,vy,dmydx,dmydy)
    return vx,vy,vz,ideg,Omdeg,om,dmxdx,dmxdy,dmydx,dmydy
