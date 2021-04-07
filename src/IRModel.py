from . import Curves
from OptionPricer import TreeBasedBondOptionPricer
import pandas as pd
import numpy as np


class BondOptionPricer:

    def __init__(self, data_, model_type_='V', intp_='CS', opt_=[0.567, 6, 8], bond_ = [0, 8, 8, 1], optCall=True, step_=1 / 12,window_len_=96):
        '''
        paras:
                @opt_: option data, opt[0]: strike; opt[1]: opt maturity; opt[2]: bond maturity
                @bond_: bond data, bond_[0]: bondCpn with 1 as unit; bond_[1]: bond TTM;
                                   bond_[2]:Time to first bond payment that is on or after option maturity,
                                            if there's a bond pmt on option maturity, it's equal to option TTM
                                            if zero bond or no cpn pmt after option maturity, it's equal to bond TTM
                                   bond_[3]: bond FaceValue
                @optCall: is call option or not
                @data_: spot curve data frame
                @model_: name of interest model, one of ['BDT','V','HW1']
                @step_: time interval for tree models or MC
                @intp_: interpolation method: one of ['CS','NS'] Cubic Spline and Nelson Siegel
        '''

        self.opt = opt_
        self.bond=bond_
        self.optCall=optCall
        self.spot_df = cls.spotBootstrapping(data_)
        self.step = step_
        self.window_len = window_len_
        self.model_type = model_type_

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
                print(x[0], e)
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
        window_len=self.window_len
        tenor = [
            i * self.step for i in range(int(np.ceil(self.opt[2] / self.step)))]
        f = lambda x: Curves.SpotRateCurve(self.spot_df.columns, x)(tenor)
        df = self.spot_df.apply(f, axis=1, result_type='expand')
        df.columns = tenor
        self.spot_df = df
        return pd.DataFrame(zip(df.columns, np.log(df[-window_len:] / df[-window_len:].shift(1)).std(axis=0)), columns=['tenor', 'vol'])

    def calibrateIRModel(self):
        # propare inputs
        if self.model_type == 'V':
            pass

        elif self.model_type == 'BDT':

            # prepare inputs
            vol = self.volStructure()
            tenor = np.array(vol['tenor'])
            vol_curve = np.array(vol['vol'])[1:]
            rates = self.spot_df.iloc[-1,:].values
            model = BDT(tenor, rates, vol_curve, step=self.step, opt = self.opt)
            model.calibrate()
            model.generateStatePrice()
            self.model = model
            # create, calibrate and save IR model
            #pass
        else:
            raise Exception('currently not support %s' % self.model_type)

    def getOptionPrice(self,showAllStates=False):
        if self.model_type= = 'BDT':
            pricer=TreeBasedBondOptionPricer(self.opt[0],self.opt[1],
                                             self.bond[0], self.bond[1],self.bond[2],self.bond[3],
                                             self.optCall, self.model)
            pricer.price(showAllStates=showAllStates)
           
        else:
            pass
            
            
            


class IRModel:

    def __init__(self, x_, y_, vol_, opt_, arg=None):
        self.arg = arg
        self.x = x_
        self.y = y_
        self.vol = vol_
        self.opt = opt_



class Vasicek(IRModel):

    def __init__(self):
        super().__init__()

    def __call__(self):

        pass

    def calibrate(self, init_arg):
        # for vasicek, 3 parameters should be determined
        #
        B = (1 - np.exp(-a * tao)) / a
        A = np.exp((B - T + tao) * (a**2 * b - sig**2 / 2) /
                   a**2 - sig**2 * B**2 / a / 4)

        pass

    @classmethod
    def analytic(cls):
        pass


if __name__ == '__main__':
    # K = 0.57, T_O = 6, T_B = 8
    pass
