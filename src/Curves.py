import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import re
from scipy.optimize import newton, brentq


class Curve:

    def __init__(self, a, *args):
        if len(a) == 0 and len(args) == 0:
            self.x = np.array([])
            self.y = np.array([])
        if len(a) != 0 and len(args) == 0:
            self.x = np.array([i[0] for i in a])
            self.y = np.array([i[1] for i in a])
        if len(args) == 1:
            self.x = np.array(a)
            self.y = np.array(args[0])
        if type(a) == str:
            df = pd.read_csv(a)
            self.x = np.array(df['x'])
            self.y = np.array(df['y'])

    def set_curve(self, x, y):
        self.x = x
        self.y = y

    def interpolate(self, new_x):
        f = interp1d(self.x, self.y, kind='linear', fill_value='extrapolate')
        return f(new_x)


class ForwardCurve (Curve):

    def __init__(self, x, y):
        super().__init__(x, y)

    def __call__(self, date):
        f = interp1d(self.x, self.y, kind='previous', fill_value='extrapolate')
        return f(date)

    def generateSpotCurve(self):
        spot_x = self.x
        spotrate_ = self.y[0]
        spotrate = [spotrate_]
        for i in range(1, len(self.y)):
            spotrate_ = (
                spotrate_ * self.x[i - 1] + self.y[i] * (self.x[i] - self.x[i - 1])) / self.x[i]
            spotrate.append(spotrate_)
        return SpotRateCurve(spot_x, spotrate)


class SpotRateCurve (Curve):

    def __init__(self, x, y):
        super().__init__(x, y)

    def __call__(self, date):
        f = interp1d(self.x, self.y, kind='cubic', fill_value='extrapolate')
        return f(date)

    def generateForwardCurve(self):
        forwardrate = [self.y[0]]
        for i in range(1, len(self.y)):
            forwardrate_ = (self.x[i] * self.y[i] - self.x[i - 1]
                            * self.y[i - 1]) / (self.x[i] - self.x[i - 1])
            forwardrate.append(forwardrate_)
        return ForwardCurve(self.x, forwardrate)


class TsyYldCurve (Curve):

    def __init__(self, x_, y_):

        # # get the most recent date
        # tre_df = pd.read_html("https://www.treasury.gov/resource-center/data-chart-center/interest-rates/Pages/TextView.aspx?data=yield")[1]
        # most_recent = tre_df[-1:].copy()
        # most_recent['Date'] = pd.to_datetime(most_recent['Date'])
        # x_base = []
        # for i in most_recent.columns[1:]:
        #     if "mo" in i:
        #         x_base.append(eval(re.findall(r'\d+',i)[0])/12)
        #     else:
        #         x_base.append(eval(re.findall(r'\d+',i)[0]))
        # y_base = list(most_recent.iloc[0][1:])
        # date_base = most_recent.iloc[0]['Date']
        # super().__init__(x_base,y_base)
        # self.curveDate = date_base

        # set specific data
        super().__init__(x_, y_)

    def getForwardCurve(self):
        forward_rate = [self.y[0] * 0.01]
        dt = np.diff(self.x, prepend=0)
        discount_factor = [1, np.exp(-self.x[0] * self.y[0] * 0.01)]
        for i in range(1, len(self.y)):
            pv = discount_factor[i] * self.y[i] * 0.01
            summation = 0
            for j in range(0, i):
                summation += discount_factor[j] * self.y[i] * 0.01 * \
                    (1 - np.exp(-forward_rate[j] * dt[j])) / forward_rate[j]
            func = lambda x: pv * \
                (1 - np.exp(-x * dt[i])) / x + summation + \
                np.exp(-x * dt[i]) * discount_factor[i] - 1
            # r = newton(func, 0.01)
            r = brentq(func, .0000001, .2)
            forward_rate.append(r)
            discount_factor.append(discount_factor[-1] * np.exp(-r * dt[i]))
        return ForwardCurve(self.x, np.array(forward_rate) * 100)

    @staticmethod
    def getMaturityInYears(x):
        if "m" in x:
            return eval(re.findall(r'\d+', x)[0]) / 12
        else:
            return eval(re.findall(r'\d+', x)[0])

    @classmethod
    def getCurveDate(cls):
        return cls.curveDate

if __name__ == '__main__':
    # print("The bootstrapped forward rate curve is\n {}".format(np.array((TsyYldCurve().getForwardCurve().x,TsyYldCurve().getForwardCurve().y))))
    # print("The spot rate curve is\n {}".format(np.array((TsyYldCurve().getForwardCurve().generateSpotCurve().x,TsyYldCurve().getForwardCurve().generateSpotCurve().y))))
    CMT_df = pd.read_csv('../data/CMT/CMT/CMT.csv')
    par = CMT_df.values[0, 1:]
    tenor = [TsyYldCurve.getMaturityInYears(x) for x in CMT_df.columns[1:]]
    par_curve = TsyYldCurve(tenor, par)
    spot_curve = par_curve.getForwardCurve().generateSpotCurve()
    print(par)
    print(tenor)
    print(spot_curve.y)
