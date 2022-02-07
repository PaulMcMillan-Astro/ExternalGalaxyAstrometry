import numpy as np

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