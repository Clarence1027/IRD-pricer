from . import Curves
from .OptionPricer import TreeBasedBondOptionPricer
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from numpy import random


class BondOptionPricer:

    def __init__(self, data, model_type='V', intp='cubic', opt=[0.57, 6, 8], bond=[0, 8, 8, 1], optCall=True, step=1 / 12, rounds=10000, window_len=96):
        '''
        paras:
                @opt: option data, opt[0]: strike; opt[1]: opt maturity; opt[2]: bond maturity
                @data: spot curve data frame
                @bond: bond data, bond_[0]: bondCpn with 1 as unit; bond_[1]: bond TTM;
                                   bond_[2]:Time to first bond payment that is on or after option maturity,
                                            if there's a bond pmt on option maturity, it's equal to option TTM
                                            if zero bond or no cpn pmt after option maturity, it's equal to bond TTM
                                   bond_[3]: bond FaceValue
                @optCall: is call option or not
                @model_: name of interest model, one of ['BDT','V','HW1']
                @step: time interval for tree models or MC
                @intp: interpolation method: one of ['CS','NS'] Cubic Spline and Nelson Siegel
        '''

        self.opt = opt
        self.bond = bond
        self.optCall = optCall
        self.spot_df = BondOptionPricer.spotBootstrapping(data)
        self.step = step
        self.intp = intp
        self.window_len = window_len
        self.model_type = model_type
        self.rounds = rounds

    @classmethod
    def spotBootstrapping(cls, df):
        tenor = [Curves.TsyYldCurve.getMaturityInYears(
            x) for x in df.columns[1:]]
        fail_dates = []

        def spot_cooking_from_df(x):
            par_curve = Curves.TsyYldCurve(tenor, x[1:])
            try:
                spot_curve = par_curve.getForwardCurve().generateSpotCurve()
                res = list(spot_curve.y)
                (res).insert(0, x[0])
            except Exception as e:
                # print(x[0], e)
                fail_dates.append(x[0])
                return None
            return res
        spot = df.apply(spot_cooking_from_df, axis=1,
                        result_type='broadcast').dropna()
        spot = spot.set_index('DATE')
        spot.columns = tenor
        return spot

    def volStructure(self):

        # interpolate spot curve with new tenor
        window_len = self.window_len
        tenor = [
            (i + 1) * self.step for i in range(int(np.ceil(self.opt[2] / self.step)))]
        f = lambda x: Curves.SpotRateCurve(
            self.spot_df.columns, x)(tenor, method=self.intp)
        df = self.spot_df.apply(f, axis=1, result_type='expand')
        df.columns = tenor
        # df.to_csv('df.csv')
        self.spot_df = df
        return pd.DataFrame(zip(df.columns, np.log(df[-window_len:] /
                                                   df[-window_len:].shift(1)).std(axis=0) / np.sqrt(5 / 252)), columns=['tenor', 'vol'])

    def calibrateIRModel(self):
        # propare inputs
        if self.model_type == 'V':
            # prepare inputs
            tenor = np.array(self.spot_df.columns)
            rates = self.spot_df.iloc[-1, :].values / 100
            rates = Curves.SpotRateCurve(tenor, rates)(tenor, method=self.intp)
            model = Vasicek(tenor, rates, vol=None, opt=self.opt)
            self.model = model

        elif self.model_type == 'BDT':

            # prepare inputs
            vol = self.volStructure()
            vol.to_csv("vol.csv")
            tenor = np.array(vol['tenor'])
            vol_curve = np.array(vol['vol'])[1:]
            rates = self.spot_df.iloc[-1, :].values / 100
            pd.DataFrame([tenor, vol_curve, rates]).to_csv('bdt_inputs.csv')
            model = BDT(tenor, rates, vol_curve,
                        step=self.step, opt=self.opt)
            # print(rates)
            # print(tenor)
            # print(vol_curve)
            model.calibrate()
            model.generateStatePrice()
            self.model = model
            # create, calibrate and save IR model
            # pass
        else:
            raise Exception('currently not support %s' % self.model_type)

    def getOptionPrice(self):
        m = self.model_type
        self.calibrateIRModel()
        if m == 'V':
            print(f"Vasicek model parameters are {self.model.arg}")
            print(f"Analytic solution for bond call {self.opt} is {self.model('analytic')}")
            print(f"MC simulation solution for bond call {self.opt} with parameters {1/self.step,self.rounds} is {self.model('MC')}")
        elif m == 'BDT':
            pricer = TreeBasedBondOptionPricer(self.opt[0], self.opt[1], self.bond[0], self.bond[
                                               1], self.bond[2], self.bond[3], self.optCall, self.model)
            print(pricer.price(showAllStates=True))
            return pricer.price()
        else:
            pass


class IRModel:

    def __init__(self, x, y, vol, opt):
        self.arg = None
        self.x = x
        self.y = y
        self.vol = vol
        self.opt = opt


class Vasicek(IRModel):

    def __init__(self, x, y, vol, opt):
        super().__init__(x, y, vol, opt)
        self.calibrate()

    def __call__(self, method='analytic', arg=[1 / 12, 1000]):
        '''
        @arg:
            arg[0]: step
            arg[1]: simulation rounds
        '''
        if method == 'analytic':
            return self.optionPrice()
        else:
            return self.MC(arg[0], arg[1])

    def P(self, arg, tao):
        r0 = self.y[1]
        a, b, sig = arg[0], arg[1], arg[2]
        B = (1 - np.exp(-a * tao)) / a
        A = np.exp((B - tao) * (a**2 * b - sig**2 / 2) /
                   a**2 - sig**2 * B**2 / a / 4)
        return A * np.exp(-B * r0)

    def calibrate(self):
        # for vasicek, 3 parameters should be determined
        def func(arg):
            tmp = 0
            for i in range(len(self.x)):
                tao = self.x[i]
                y = self.y[i]
                tmp += (self.P(arg, tao) - np.exp(-tao * y))**2
            return tmp
        res = minimize(func, (0.1, 0.1, 0.1), bounds=(
            (None, None), (None, None), (0.02, 1)))
        print(res)
        self.arg = res.x

    def optionPrice(self):
        r0 = self.y[1]
        arg = self.arg
        a, b, sig = arg[0], arg[1], arg[2]
        X = self.opt[0]
        tao = self.opt[2]
        tao_ = self.opt[1]
        vol = sig / a * (1 - np.exp(-a * (tao - tao_))) * \
            np.sqrt((1 - np.exp(-2 * a * tao_)) / 2 / a)
        d = np.log(self.P(arg, tao) / X * self.P(arg, tao_)) / vol + vol / 2
        return self.P(arg, tao) * norm.cdf(d) - X * self.P(arg, tao_) * norm.cdf(d - vol)

    def MC(self, step=1 / 12, rounds=10000):
        deltaT = step
        r0 = self.y[1]
        K = self.opt[0]
        arg = self.arg
        a, b, sig = arg[0], arg[1], arg[2]
        tao = self.opt[2]
        tao_ = self.opt[1]
        N = int(tao / deltaT)
        n = int(tao_ / deltaT)

        res = []
        for i in range(rounds):
            rand = random.normal(0, 1, size=N)

            def f(r, x):
                return r * np.exp(-a * deltaT) + b * (1 - np.exp(-a * deltaT)) + \
                    np.sqrt(sig**2 * (1 - np.exp(-2 * a * deltaT)) / 2 / a) * x
            r = r0
            path = []
            for z in rand:
                r_ = f(r, z)
                path.append(r_)
                r = r_
            path = np.array(path)
            res.append(np.exp(-(path[:n] * deltaT).sum()) *
                       max(np.exp(-(path[n:] * deltaT).sum()) - K, 0))
        return np.mean(res)


class BDT(IRModel):

    def __init__(self, x_, y_, vol_, step, opt=None):
        '''
        :param step: step size of time evolution
        '''
        super().__init__(x_, y_, vol_, opt)
        self.step = step

    def calibrate(self):

        if len(self.x) != len(self.y) or len(self.x) != len(self.vol) + 1:
            raise Exception('length does not match')
        else:
            zeroPrice = np.exp(-self.y * self.x)

            initGuess = np.vstack((self.y[1:], self.vol)).T
            bounds = np.array([((0, 0.1), (0, 4))
                               for i in range(len(self.vol))])
            bounds = [list(x) for x in bounds.reshape(len(self.vol) * 2, 2)]

            res = minimize(BDT.totalSSE,
                           initGuess,
                           args=(self.step, zeroPrice, self.vol),
                           method='SLSQP', bounds=bounds)

            if res.success:
                #print('So damn good ')
                # out = res.x.reshape(-1, 2)
                out = res.x.reshape(len(self.vol), 2)
                shortRate = np.triu(np.ones((len(zeroPrice), len(zeroPrice))))
                shortRate[0, 0] = -np.log(zeroPrice[0]) / self.step
                for i in range(1, len(shortRate)):
                    shortRate[0:i + 1, i] = [out[i - 1, 0] * (
                        np.exp(2 * out[i - 1, 1] * np.sqrt(self.step)))**k for k in range(i + 1)]

                self.shortRate = shortRate

            else:
                # print('Fuckkkkk')
                print(res)
                print(initGuess)
                raise Exception('optimization failed')

    @staticmethod
    def SSE(shortRateTree, deltaTime, timeIndex, targetPrice, targetVol):
        # calculate error for timeIndex using the structed shortRateTree

        priceTree = np.triu(np.ones((timeIndex + 2, timeIndex + 2)))

        for t in range(timeIndex, -1, -1):
            for j in range(0, t + 1):
                priceTree[j, t] = 0.5 * (priceTree[j, t + 1] + priceTree[
                                         j + 1, t + 1]) * np.exp(-shortRateTree[j, t] * deltaTime)

        sigma_model = 0.5 * \
            np.log(np.log(priceTree[1, 1]) / np.log(priceTree[0, 1]))

        return 10 * ((priceTree[0, 0] - targetPrice) ** 2 + (sigma_model - targetVol) ** 2)

    @staticmethod
    def totalSSE(xx, deltaTime, targetPrices, targetVols):
        # xx= [[r_10, sigma1],
        #      [r_20, sigma2],
        #      [r_30, sigma3]
        #       .............
        #                    ]
        xx = xx.reshape(-1, 2)

        totalError = 0

        shortRateTree = np.triu(
            np.zeros((len(targetPrices), len(targetPrices))))
        shortRateTree[0, 0] = -np.log(targetPrices[0]) / deltaTime

        for i in range(1, len(shortRateTree)):
            shortRateTree[0:i + 1, i] = [xx[i - 1, 0] *
                                         (np.exp(2 * xx[i - 1, 1] * np.sqrt(deltaTime)))**k for k in range(i + 1)]

        for i in range(1, len(shortRateTree)):
            totalError += BDT.SSE(shortRateTree, deltaTime,
                                  i, targetPrices[i], targetVols[i - 1])

        return totalError

    def generateStatePrice(self):
        self.statePrice = [None]
        for i in range(1, len(self.shortRate) + 1):
            px = np.triu(np.ones((i + 1, i + 1)))
            for t in range(i - 1, -1, -1):
                for j in range(0, t + 1):
                    px[j, t] = 0.5 * (px[j, t + 1] + px[j + 1, t + 1]) * \
                        np.exp(-self.shortRate[j, t] * self.step)

            self.statePrice.append(px)


def BDT_testDemo():

    vol = np.array([0.396985268, 0.300462881,
                    0.240958353, 0.205686601, 0.182102897, 0.163445286,
                    0.148452631, 0.137814264, 0.131999614, 0.12995787,
                    0.129306795, 0.128412379, 0.126670036, 0.124053075,
                    0.120846305, 0.117463875, 0.114323428, 0.111757181,
                    0.10995445, 0.108939193, 0.108585538, 0.108665657,
                    0.108925065, 0.109193383, 0.10940044, 0.109538009,
                    0.10963047, 0.109713755, 0.109821133, 0.109974592,
                    0.110180777, 0.110430573, 0.11070143, 0.110961576,
                    0.111177119, 0.11132523, 0.111394387, 0.111380694,
                    0.111285541, 0.111113778, 0.110872306, 0.110569007,
                    0.110211948, 0.10980881, 0.109366493, 0.108890879,
                    0.108386702, 0.107857528, 0.107305803, 0.106732966,
                    0.106139607, 0.105525649, 0.104890556, 0.104233547,
                    0.103553806, 0.102850688, 0.102123911, 0.101373729,
                    0.100602227])

    rates = np.array([0.0001, 0.000127636, 0.000200001, 0.00029855,
                      0.000404738, 0.00050002, 0.000569998, 0.000616866,
                      0.000646964, 0.000666633, 0.000682215, 0.000700049,
                      0.000725379, 0.000759056, 0.000800834, 0.000850465,
                      0.000907703, 0.000972302, 0.001044015, 0.001122595,
                      0.001207795, 0.001299369, 0.001397071, 0.001500653,
                      0.001609923, 0.001724907, 0.001845683, 0.00197233,
                      0.002104929, 0.002243557, 0.002388295, 0.002539221,
                      0.002696414, 0.002859955, 0.003029921, 0.003206393,
                      0.003389376, 0.00357858, 0.003773641, 0.003974196,
                      0.004179881, 0.004390333, 0.004605189, 0.004824083,
                      0.005046654, 0.005272538, 0.00550137, 0.005732788,
                      0.005966428, 0.006201926, 0.006438919, 0.006677043,
                      0.006915934, 0.00715523, 0.007394566, 0.007633579,
                      0.007871906, 0.008109182, 0.008345045, 0.00857913])

    ttm = np.arange(1, 61) * (1 / 12)

    '''
    ttm=np.arange(1,5)*0.25
    px=np.array([0.9888,0.9775,0.9664,0.9555])
    rates=-np.log(px)/ttm
    vol=np.array([44.48,39.41,37.52])/100
    '''

    bdt = BDT(ttm, rates, vol, step=1 / 12)
    bdt.calibrate()
    # print(bdt.shortRate)
    # print('------------')
    bdt.generateStatePrice()
    # for tree in bdt.statePrice:
    print(tree)
    print('----------')


if __name__ == '__main__':

    BDT_testDemo()

    # # test runing example
    # # using command line: python -m src.IRModel under project ./ dir
    # data = pd.read_csv('./data/CMT/CMT/CMT.csv')
    # model_type = 'BDT'
    # intp = 'cubic'
    # opt = [0.57, 6, 8]
    # bond = [0, 8, 8, 1]
    # optCall = True
    # step = 1 / 12
    # rounds = 10000
    # window_len = 96
    # pricer = BondOptionPricer(data, model_type, intp,
    #                           opt, bond, optCall, step, rounds, window_len)
    # pricer.getOptionPrice()
