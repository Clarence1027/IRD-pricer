import numpy as np
from scipy.optimize import minimize

class ShortRate():
    def __init__(self,arg,x_,y_,vol_):
        self.arg=arg
        self.x=x_ # ttm, np.array
        self.y=y_ # zero rates, np.array
        self.vol=vol_ # vol, np.array


class BDT(ShortRate):
    def __init__(self,x_,y_,vol_,step,arg=None):
        '''
        :param step: step size of time evolution
        '''
        super().__init__(arg,x_,y_,vol_)
        self.step=step
    
    
    def calibrate(self):

        if len(self.x) != len(self.y) or len(self.x) != len(self.vol)+1:
            raise Exception('length does not match')
        else:
            zeroPrice=np.exp(-self.y*self.x)
            self.shortRate=np.triu(np.zeros((len(self.x),len(self.x))))
            priceTree=np.triu(np.ones((len(self.x)+1,len(self.x)+1))) # an auxiliary tree
            for i in range(len(self.shortRate)):
                # to calibrate short rates at step i
                if i == 0:
                    self.shortRate[0,0]=-np.log(zeroPrice[0])/self.step

                else:
                    res=minimize(BDT.SSE,[0.01,0.1],args=(self.shortRate,priceTree,self.step,i,zeroPrice[i],self.vol[i-1]),
                                 method='SLSQP')
                    if res.success:
                        pass
                    else:
                        raise Exception('optimization failed for step %d' % i)


    @staticmethod
    def SSE(x,shortRateTree,priceTree,deltaTime,timeIndex,targetPrice,targetVol):
        # x=[r_i0,sigma] where i=timeIndex, r_i0 = (0,i) element in shortRateTree
        r,sigma=x[0],x[1]
        shortRateTree[0:timeIndex+1,timeIndex]=[ r*np.exp(2*sigma*np.sqrt(deltaTime))**k for k in range(timeIndex+1) ]
        for t in range(timeIndex,-1,-1):
            for j in range(0,t+1):
                priceTree[j,t]=0.5*(priceTree[j,t+1]+priceTree[j+1,t+1])*np.exp(-shortRateTree[j,t]*deltaTime)

        sigma_model=0.5*np.log(np.log(priceTree[1,1])/np.log(priceTree[0,1]))

        return 100000*((priceTree[0,0]-targetPrice)**2 + (sigma_model-targetVol)**2)

    def generateStatePrice(self):
        self.statePrice=[None]
        for i in range(1,len(self.shortRate)+1):
            px=np.triu(np.ones((i+1,i+1)))
            for t in range(i-1,-1,-1):
                for j in range(0,t+1):
                    px[j,t]=0.5*(px[j,t+1]+px[j+1,t+1])*np.exp(-self.shortRate[j,t]*self.step)

            self.statePrice.append(px)


'''
def testDemo():
    ttm=np.arange(1,21)*0.25

    px=np.array([0.9888,0.9775,0.9664,0.9555,
                 0.9447,0.9341,0.9235,0.9131,
                 0.9028,0.8926,0.8826,0.8726,
                 0.8628,0.8530,0.8434,0.8339,
                 0.8245,0.8152,0.8060,0.7969])

    rates=-np.log(px)/ttm

    vol=np.array([44.48,39.41,37.52,36.59,
                   36.02,35.58,35.15,34.70,
                   34.19,33.62,33.01,32.36,
                   31.68,31.00,30.31,29.65,
                   29.01,28.41,27.87])/100

    bdt=BDT(ttm,rates,vol,step=0.25)
    bdt.calibrate()
    bdt.generateStatePrice()

    print(bdt.shortRate[:4,:4])
    print('------------')

    for i in range(1,5):
        print(bdt.statePrice[i])
        print('------------')

'''