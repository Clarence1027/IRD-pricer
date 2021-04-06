import numpy as np
from scipy.optimize import minimize
from . import IRModel


class BDT(IRModel.IRModel):

    def __init__(self, x_, y_, vol_, opt_=None):
        super().__init__(x_, y_, vol_, opt_)

    def calibrate(self):

        if len(self.x) != len(self.y) or len(self.x) != len(self.vol) + 1:
            raise Exception('length does not match')
        else:
            zeroPrice = np.exp(-self.y * self.x)
            deltaTime = np.diff(self.x, prepend=0)
            self.shortRate = np.triu(np.zeros((len(self.x), len(self.x))))
            self.priceTree = np.triu(
                np.ones((len(self.x) + 1, len(self.x) + 1)))
            for i in range(len(self.shortRate)):
                # minimize(obj,(r,sigma),args)
                # fill in rate
                # to calibrate short rates at step i
                if i == 0:
                    self.shortRate[0, 0] = -np.log(zeroPrice[0]) / deltaTime[0]

                else:
                    res = minimize(BDT.SSE, [0.01, 0.1], args=(
                        self.shortRate, self.priceTree, deltaTime, i, zeroPrice[i], self.vol[i - 1]))
                    if res.success:
                        self.shortRate[0:i + 1, i] = [res.x[0] *
                                                      np.exp(2 * res.x[1])**k for k in range(i + 1)]
                    else:
                        raise Exception('optimization failed for step %d' % i)

    @staticmethod
    def SSE(x, shortRateTree, priceTree, deltaTime, timeIndex, targetPrice, targetVol):
        # x=[r_i0,sigma] where i=timeIndex, r_i0 = (0,i) element in
        # shortRateTree
        r, sigma = x[0], x[1]
        shortRateTree[0:timeIndex + 1, timeIndex] = [r *
                                                     np.exp(2 * sigma)**k for k in range(timeIndex + 1)]
        for t in range(timeIndex, -1, -1):
            for j in range(0, t + 1):
                priceTree[j, t] = 0.5 * (priceTree[j, t + 1] + priceTree[
                                         j + 1, t + 1]) * np.exp(-shortRateTree[j, t] * deltaTime[t])

        sigma_model = 0.5 * \
            np.log(np.log(priceTree[1, 1]) / np.log(priceTree[0, 1]))

        return 100000 * ((priceTree[0, 0] - targetPrice)**2 + (sigma_model - targetVol)**2)


def usageDemo():
    ttm = np.array([1, 2, 3])
    vol = np.array([19, 18]) / 100
    px = np.array([0.9091, 0.8116, 0.7118])
    rates = -np.log(px) / ttm

    bdt = BDT(ttm, rates, vol)

    bdt.calibrate()

    print(bdt.shortRate)
    print(bdt.priceTree)
if __name__ == '__main__':
    usageDemo()
