

import random
from lagrange import lagrange

class Lagrange(object):
    
    LIMPRIME = 251
    DEGREE = 2
    
    @staticmethod
    def interpolate(values):
        dct={}
        for i in range(0,len(values)):
            dct[i+1] = values[i]
        r = lagrange(dct,Lagrange.LIMPRIME)
        return r
    
    def __init__(self,secret):
        self.coeffs = [random.randint(1,Lagrange.LIMPRIME-1) for i in range(0,Lagrange.DEGREE-1)]
        self.secret = secret
        
    def secretBuild(self,x):
        s = self.secret
        for i in range(0,len(self.coeffs)):
            s = s + pow(x,i+1)*self.coeffs[i]
        return int(s % Lagrange.LIMPRIME)
    
    def secretShare(self,nservers):
        
        return [self.secretBuild(i+1) for i in range(0,nservers)]
    
      
    
        
            
        
        
        
        
        
        
        