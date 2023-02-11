from math import log

from gym.spaces import Tuple,Discrete,Box,Dict,MultiDiscrete,MultiBinary
def weight_normalizer():
    a=Box(0, 1, shape=(3,)).sample()
    a=a/sum(a)
    return a

def points_initializer(points):
    a = Box(0, 100, shape=(3,1)).sample()
    a[0][0]=points.values[0]
    #a[0][1]=points.values[1]
    a[1][0] =points.values[2]
    #a[1][1] =points.values[3]
    a[2][0] =points.values[4]
    #a[2][1] =points.values[5]
    return a




def state_initializer(points,weights):
    a=Box(0, 100, shape=(1,1)).sample()
    #a[0][0]=((points[0][0]*weights[0])/1.70394 +(points[1][0]*weights[1])/1.43955+ (points[2][0]*weights[2])/0.483042)/3.3514745
    a[0][0]=(points[0][0]*weights[0]+points[1][0]*weights[1]+points[2][0]*weights[2])
    return a

def fisher_information(points,state):
    a = Box(0, 1, shape=(1, 1)).sample()

    #####"##### this us the fisher information :
    a[0][0]=((points[0][0]-state[0][0])/1.70394)**2+((points[1][0]-state[0][0])/1.43955)**2+((points[2][0]-state[0][0])/0.483042)**2
    #a[0][0] = ((points[0][0]-state[0][0])/5.9)**2 + ((points[0][1]-state[0][1])/10.5)**2
    #a[0][1] = ((points[1][0]-state[0][0])/1.49)**2 + ((points[1][1]-state[0][1])/1.007)**2+((points[2][0]-state[0][0])/0.6196)**2
    #a[0][2] =((points[2][0]-state[0][0])/12.9137)**2 + ((points[2][1]-state[0][1])/17.9028)**2

    ###########this is the entropy one :
    #a[0][0]=(points[0][0]*weights[0]/(points[0][0]*weights[0]+points[1][0]*weights[1]+points[2][0]*weights[2]))*log((points[0][0]*weights[0]/(points[0][0]*weights[0]+points[1][0]*weights[1]+points[2][0]*weights[2])))
    #a[1][0]=(points[1][0]*weights[1]/(points[0][0]*weights[0]+points[1][0]*weights[1]+points[2][0]*weights[2]))*log((points[1][0]*weights[1]/(points[0][0]*weights[0]+points[1][0]*weights[1]+points[2][0]*weights[2])))
    #a[2][0]=(points[2][0]*weights[2]/(points[0][0]*weights[0]+points[1][0]*weights[1]+points[2][0]*weights[2]))*log((points[2][0]*weights[2]/(points[0][0]*weights[0]+points[1][0]*weights[1]+points[2][0]*weights[2])))
    results=a[0][0]
    return results
