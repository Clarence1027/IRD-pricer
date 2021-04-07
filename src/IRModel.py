from . import Curves
from . import BDTmodel
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
            i * self.step for i in range(int(np.ceil(self.opt[2] / self.step)))]
        f = lambda x: Curves.SpotRateCurve(
            self.spot_df.columns, x)(tenor, method=self.intp)
        df = self.spot_df.apply(f, axis=1, result_type='expand')
        df.columns = tenor
        self.spot_df = df
        return pd.DataFrame(zip(df.columns, np.log(df[-window_len:] /
                                                   df[-window_len:].shift(1)).std(axis=0)), columns=['tenor', 'vol'])

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
            tenor = np.array(vol['tenor'])
            vol_curve = np.array(vol['vol'])[1:]
            rates = self.spot_df.iloc[-1, :].values / 100
            model = BDTmodel.BDT(tenor, rates, vol_curve,
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
            pricer.price(showAllStates=showAllStates)
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


if __name__ == '__main__':

    # test runing example
    # using command line: python -m src.IRModel under project ./ dir
    data = pd.read_csv('./data/CMT/CMT/CMT.csv')
    model_type = 'BDT'
    intp = 'cubic'
    opt = [0.57, 6, 8]
    bond = [0, 8, 8, 1]
    optCall = True
    step = 1 / 12
    rounds = 10000
    window_len = 96
    pricer = BondOptionPricer(data, model_type, intp,
                              opt, bond, optCall, step, rounds, window_len)
    pricer.getOptionPrice()
