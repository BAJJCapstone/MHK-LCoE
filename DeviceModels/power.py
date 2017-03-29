from scipy.integrate import odeint
import numpy as np


class Turbine:
    def __init__(self, K, Q, B, M, g):
        '''
        K: Upper asymptote set to the installed capacity kW
        Q: Depends on turbine design
        B: Rate of power increase
        M: Maximum increase
        g: symmetry of increase
        '''
        self.K = K
        self.Q = Q
        self.B = B
        self.M = M
        self.g = g

    def richardsCurve(self, Velocity):
        '''
        Population curve model applied to power curve of turbines
        '''
        #print(Velocity)
        return self.K * (1+self.Q*np.exp(-1*self.B*(Velocity-self.M)))**(-1/self.g)

    def instantaneousPower(self, P, time, HM):
        return self.richardsCurve(abs(harmonicConstituentModel(time, *HM)))

    def instantaneousPowerWithStation(self, P, time, TidalStation, gravity, height):
        '''
        Calculates the power given the current velocity

        '''
        return self.richardsCurve(abs(TidalStation.velocityFromConstituent(time/3600, gravity, height)))

def harmonicConstituentModel(time, *hm):
    assert len(hm) % 3 == 0
    velocity = 0
    for i in range(len(hm)//3):
        velocity += hm[3*i]*np.sin((hm[3*i+1] * time + hm[3*i+2])*np.pi/180.)
    return velocity

def calculate_power(HM, Turbine, p_0, time_start, time_end):
    '''
    Integrate the big equation to determine how much power has been generated
    '''
    times = np.arange(time_start, time_end)
    result = odeint(Turbine.instantaneousPower, p_0, times, args=(HM,))
    return result, times
