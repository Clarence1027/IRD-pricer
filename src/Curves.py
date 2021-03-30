
import numpy
from scipy import interpolate as ip
import pandas as pd
import math
from scipy.optimize import brentq

class Curve():
    
    
    def __init__(self,x_ = None, y_ = None):
        self.set_curve(x_,y_)
        
        
        
    def set_curve(self, x_ = None, y_ = None):

        if type(x_) is list and y_ is None:
            self.x = numpy.array([coor[0] for coor in x_])
            self.y = numpy.array([coor[1] for coor in x_])
            
        elif type(x_) is list and type(y_) is list:
            self.x = numpy.array(x_)
            self.y = numpy.array(y_) 
            
        elif type(x_) is numpy.ndarray and type(y_) is numpy.ndarray:
            self.x = x_
            self.y = y_
            
        elif type(x_) is str:
            csv = numpy.genfromtxt(x_, delimeter = ",")
            self.x = csv[:,0]
            self.y = csv[:,1]
            
        else:
            self.x = numpy.array([])
            self.y = numpy.array([])
            
            
    def interpolate(self, opt = 'linear', xNew = None):
        if xNew is not None:
            return ip.interp1d(self.x, self.y, kind = opt, fill_value="extrapolate")(xNew)
        else:
            return ip.interp1d(self.x,self.y, kind = opt, fill_value="extrapolate")
            
class TsyYldCurve(Curve):
    
    def __init__(self, x_ = None, y_ = None, curveDate_ = None):
        self.set_date(curveDate_)
        super().__init__(x_,y_)
    
    def set_date(self, newDate):
        self.curveDate = newDate
        
    def getCurveDate(self):
        return self.curveDate
    
    @staticmethod
    def getMaturityInYears(maturityDates):
        maturityInStr = list(maturityDates)
        maturityInYear = []
        for maturity in maturityInStr:
            if 'mo' in maturity:
                maturityInYear += [float(maturity.replace(' mo', ''))/12]
            else:
                maturityInYear += [float(maturity.replace(' yr', ''))]
        return maturityInYear
    
    @staticmethod
    def buildMostRecentDate():
        dfs = pd.read_html('https://www.treasury.gov/resource-center/data-chart-center/interest-rates/Pages/TextView.aspx?data=yield')
        maturityDays = list(dfs[1].columns)[1:]
        xData = numpy.array(TsyYldCurve.getMaturityInYears(maturityDays))
        yData = numpy.array(list(dfs[1].iloc[-1,1:]))/100
        date_ = dfs[1].iloc[-1,0]
        return TsyYldCurve(xData,yData,date_)
    
    @staticmethod
    def DF(rate, time_diff):
        return numpy.exp(-rate.dot(time_diff))
   
    @staticmethod
    def couponNPV(coupon, rate, time_diff):
        discountFactor = numpy.cumprod(numpy.exp(-rate*time_diff))
        return numpy.sum((coupon/rate)*discountFactor*(1-numpy.exp(-rate*time_diff)))
    
    def generateForwardCurve(self):
        time_diff = numpy.diff(self.x, prepend = 0)
        rate = self.y.copy()
        for i in range(1,len(time_diff)):
            coupon = self.y[i]
            df_i = TsyYldCurve.DF(rate[:i],time_diff[:i])
            f = lambda a: TsyYldCurve.couponNPV(coupon, rate[:i], time_diff[:i]) + coupon/a * df_i * (1-numpy.exp(-a*time_diff[i])) + df_i*numpy.exp(-a*time_diff[i]) -1
            rate[i]= brentq(f,.0001,.1)
        return ForwardCurve(self.x, rate)
    
class ForwardCurve(Curve):
    
    def __init__(self, x_= None,y_ = None):
        super().__init__(x_,y_)
    
    def __call__(self, x_, opt = 'previous'):
        return self.interpolate(x_, opt)
    
    def generateSpotCurve(self):
        time_diff = numpy.diff(self.x, prepend = 0)
        new_y = numpy.zeros(len(time_diff))
        for i in range(len(time_diff)):
            f = lambda a: numpy.exp(a*self.x[i])-numpy.exp((self.y[:i+1]*time_diff[:i+1]).sum())
            new_y[i] = brentq(f,-.0001,.1)
        return SpotCurve(self.x, new_y)
        

class SpotCurve(Curve):
    def __init__(self, x_= None,y_ = None):
        super().__init__(x_,y_)
    
    def __call__(self, x_, opt = 'linear'):
        return self.interpolate(x_, opt)
    
    def generateForwardCurve(self):
        time_diff = numpy.diff(self.x, prepend = 0)
        new_y = numpy.zeros(len(time_diff))
        new_y[0] =self.y[0]
        for i in range(1,len(time_diff)):
            f = lambda a: math.exp(self.y[i]*self.x[i])/(math.exp(self.y[i-1]*self.x[i-1])-math.exp(a*time_diff[i]))
            new_y[i] = brentq(f,.0001,.1)
        return ForwardCurve(self.x, new_y)
            
            
            
if __name__ == '__main__':
    
    
    
   # T = [0.02, 0.04, 0.1, 0.3, 0.5, 0.75, 1, 1.5, 2, 5, 10, 15, 20, 25, 30, 50]
    
    recentYldCurve = TsyYldCurve.buildMostRecentDate()
    
    print('The date of the yields is from ' + recentYldCurve.curveDate)
    [ print(str(recentYldCurve.y[t]) + ' for ' + str(recentYldCurve.x[t]) + ' years') for t in range(len(recentYldCurve.x))]
    print("\n")
    print('The forward rates are: ')
    fwd = recentYldCurve.generateForwardCurve()
    spt = fwd.generateSpotCurve()
    [ print(str(fwd.y[t]) + ' for ' + str(fwd.x[t]) + ' years') for t in range(len(fwd.x))]
    print("\n")
    print('The spot rates are: ')
    [ print(str(spt.y[t]) + ' for ' + str(fwd.x[t]) + ' years') for t in range(len(spt.x))]
    