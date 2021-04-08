import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import re
from scipy.optimize import newton, brentq, minimize


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

    def __call__(self, date, method='linear'):
        # f = interp1d(self.x, self.y, kind=method, fill_value='extrapolate')
        return interpolation(self.x, self.y, method, date)

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


def interpolation(tenor, y, method, tenor_new):  # method can be either 'Cubic' or 'NS'
    if method == 'Cubic':

        # determine interval length
        n = int(np.floor(len(tenor) / 3))

        def Cubic(x, t):
            return x[0] + x[1] * t + x[2] * t**2 + x[3] * t**3

        def fun(x):
            tmp1 = np.array([Cubic(x[:4], tenor[i]) - y[i] for i in range(n)])
            tmp2 = np.array([Cubic(x[4:8], tenor[i]) - y[i]
                             for i in range(n, 2 * n)])
            tmp3 = np.array([Cubic(x[8:], tenor[i]) - y[i]
                             for i in range(2 * n, len(tenor))])

            return (tmp1**2).sum() + (tmp2**2).sum() + (tmp3**2).sum()

        def F1(x, t):  # first derivative
            return x[1] + 2 * x[2] * t + 3 * x[3] * t**2

        def F2(x, t):  # second derivative
            return 2 * x[2] + 6 * x[3] * t

        cons = ({'type': 'eq', 'fun': lambda x: Cubic(x[:4], tenor[n - 1]) - Cubic(x[4:8], tenor[n - 1])},
                {'type': 'eq', 'fun': lambda x: Cubic(
                    x[4:8], tenor[2 * n - 1]) - Cubic(x[8:], tenor[2 * n - 1])},
                {'type': 'eq', 'fun': lambda x: F1(
                    x[:4], tenor[n - 1]) - F1(x[4:8], tenor[n - 1])},
                {'type': 'eq', 'fun': lambda x: F1(
                    x[4:8], tenor[2 * n - 1]) - F1(x[8:], tenor[2 * n - 1])},
                {'type': 'eq', 'fun': lambda x: F2(
                    x[:4], tenor[n - 1]) - F2(x[4:8], tenor[n - 1])},
                {'type': 'eq', 'fun': lambda x: F2(x[4:8], tenor[2 * n - 1]) - F2(x[8:], tenor[2 * n - 1])})

        x0 = [0.5 for i in range(12)]
        res = minimize(fun, x0, constraints=cons)
        n_ = int(np.floor(len(tenor_new) / 3))
        model = [Cubic(res.x[:4], tenor_new[i]) for i in range(n_)]
        model += [Cubic(res.x[4:8], tenor_new[i]) for i in range(n_, 2 * n_)]
        model += [Cubic(res.x[8:], tenor_new[i])
                  for i in range(2 * n_, len(tenor_new))]
        if not res.success:
            return None
        return np.array(model)

    elif method == 'NS':
        n = len(tenor)

        def NS(x, t):
            exp = np.exp(-t / x[3])
            return x[0] + x[1] * ((1 - exp) / (t / x[3])) + x[2] * (((1 - exp) / (t / x[3])) - exp)

        def fun(x):
            tmp = np.array([NS(x, tenor[i]) - y[i] for i in range(n)])
            return (tmp**2).sum()

        x0 = [0.1 for i in range(4)]
        res = minimize(fun, x0)
        # print(res)
        if not res.success:
            f = interp1d(tenor, y, kind='cubic', fill_value='extrapolate')
            return f(tenor_new)
        return np.array([NS(res.x, tenor_new[i]) for i in range(len(tenor_new))])

if __name__ == '__main__':
    # print("The bootstrapped forward rate curve is\n {}".format(np.array((TsyYldCurve().getForwardCurve().x,TsyYldCurve().getForwardCurve().y))))
    # print("The spot rate curve is\n {}".format(np.array((TsyYldCurve().getForwardCurve().generateSpotCurve().x,TsyYldCurve().getForwardCurve().generateSpotCurve().y))))
    # CMT_df = pd.read_csv('../data/CMT/CMT/CMT.csv')
    # par = CMT_df.values[0, 1:]
    # tenor = [TsyYldCurve.getMaturityInYears(x) for x in CMT_df.columns[1:]]
    # par_curve = TsyYldCurve(tenor, par)
    # spot_curve = par_curve.getForwardCurve().generateSpotCurve()
    # print(par)
    # print(tenor)
    # print(spot_curve.y)

    tenor = np.array([1 / 12, 0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30])
    yields = np.array([0.04, 0.0600005, 0.110007, 0.240086, 0.691685,
                       1.15725, 2.16042, 2.86866,  3.57557,  4.57708,  4.8913])
    print(interpolation(tenor, yields, 'NS', tenor))
