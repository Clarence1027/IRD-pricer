import numpy as np


class TreeBasedBondOptionPricer():

    def __init__(self, optStrike, optTTM, bondCpn, bondTTM, TTFirstPmt, bondFV=1, optCall=True, shortRateModel=None):
        '''
        :param optStrike: strike price of option
        :param optTTM:  time to maturity of option
        :param bondCpn: bond coupon rate in decimal, i.e the unit is 1; eg: 0.05 for 5%
        :param bondTTM: time to maturity of bond
        :param TTFirstPmt: time to the first payment that is on or after option maturity
        :param bondFV: bond face value
        :param optCall: call option or not
        :param shortRateModel: object that is a short rate model which is tree based
        '''
        if optTTM > bondTTM or TTFirstPmt < optTTM or TTFirstPmt > bondTTM:
            raise Exception(
                'invalid option TTM with regard to bond TTM and TTFirstPmt')
        if (bondTTM - TTFirstPmt) / 0.5 != int((bondTTM - TTFirstPmt) / 0.5):
            raise Exception('invalid TTFirstPmt with regard to bond TTM')
        if shortRateModel is not None and (shortRateModel.step > 0.5 or 0.5 / shortRateModel.step != int(0.5 / shortRateModel.step)):
            raise Exception(
                'improper short rate model time step. It MUST be less or equal than 0.5 and is a divider of 0.5')
        if shortRateModel is not None and optTTM / shortRateModel.step != int(optTTM / shortRateModel.step):
            raise Exception(
                'improper short rate model time step for option TTM')
        if shortRateModel is not None and bondTTM / shortRateModel.step != int(bondTTM / shortRateModel.step):
            raise Exception('improper short rate model time step for bond TTM')

        self.optStrike = optStrike
        self.optTTM = optTTM
        self.bondCpn = bondCpn
        self.bondTTM = bondTTM
        self.TTFirstPmt = TTFirstPmt
        self.bondFV = bondFV
        self.optCall = optCall
        self.model = shortRateModel

    def _getOptMaturityTimeIndex(self):
        return int(self.optTTM / self.model.step)

    def _getPmtTimeIndex(self):
        firstPmtTimeIndex = self._getOptMaturityTimeIndex(
        ) + int((self.TTFirstPmt - self.optTTM) / self.model.step)
        numOfPmt = int((self.bondTTM - self.TTFirstPmt) * 2) + 1
        increment = int(0.5 / self.model.step)
        return [i * increment + firstPmtTimeIndex for i in range(numOfPmt)]

    def optPayOffAtMaturity(self):
        # for each time index from self.getPmtTimeIndex, and for given option maturity time index
        # find corresponding price tree from which find price of $1 at option maturity
        # then times $coupon,  sum all the price up to get state option payoff at maturity
        # remember finally reshape to two dimensional for later discount
        # convenience
        if self.model is None:
            raise Exception('set short rate tree model first')

        optMatTimeIndx = self._getOptMaturityTimeIndex()
        pmtTimeIndx = self._getPmtTimeIndex()
        payoff = 0
        for t in pmtTimeIndx:
            payoff += (self.model.statePrice[t][:optMatTimeIndx + 1,
                                                optMatTimeIndx] * self.bondCpn * self.bondFV)

        payoff += (self.model.statePrice[t]
                   [:optMatTimeIndx + 1, optMatTimeIndx] * self.bondFV)

        if self.optCall:
            payoff = np.maximum(payoff - self.optStrike, 0)
        else:
            payoff = np.maximum(self.optStrike - payoff, 0)

        return payoff

    def price(self, showAllStates=False):
        # with state option payoff at maturity, using short rate tree to discount back
        # if bdt, simple take average of two state and discount back
        # if Hull-White,  using prob trees for average and discount back
        payoff = self.optPayOffAtMaturity()
        optMatTimeIndx = self._getOptMaturityTimeIndex()
        optPrice = np.triu(np.ones((optMatTimeIndx + 1, optMatTimeIndx + 1)))
        optPrice[:optMatTimeIndx + 1, optMatTimeIndx] = payoff

        for t in range(optMatTimeIndx - 1, -1, -1):
            for j in range(0, t + 1):
                optPrice[j, t] = 0.5 * (optPrice[j, t + 1] + optPrice[j + 1, t + 1]) * \
                    np.exp(-self.model.shortRate[j, t] * self.model.step)

        if not showAllStates:
            return optPrice[0, 0]
        else:
            return optPrice


def demo():
    from .BDTmodel import BDT
    # bdt setup
    ttm = np.arange(1, 21) * 0.25

    px = np.array([0.9888, 0.9775, 0.9664, 0.9555,
                   0.9447, 0.9341, 0.9235, 0.9131,
                   0.9028, 0.8926, 0.8826, 0.8726,
                   0.8628, 0.8530, 0.8434, 0.8339,
                   0.8245, 0.8152, 0.8060, 0.7969])

    rates = -np.log(px) / ttm

    vol = np.array([44.48, 39.41, 37.52, 36.59,
                    36.02, 35.58, 35.15, 34.70,
                    34.19, 33.62, 33.01, 32.36,
                    31.68, 31.00, 30.31, 29.65,
                    29.01, 28.41, 27.87]) / 100

    bdt = BDT(ttm, rates, vol, step=0.25)
    bdt.calibrate()
    bdt.generateStatePrice()

    # option and bond setup for pricer
    optStrike = 0.9
    optTTM = 0.25
    bondCpn = 0.05
    bondTTM = 1
    TTFirstPmt = 0.5

    pricer = TreeBasedBondOptionPricer(
        optStrike, optTTM, bondCpn, bondTTM, TTFirstPmt, shortRateModel=bdt)

    # print(pricer._getOptMaturityTimeIndex())
    # print(pricer._getPmtTimeIndex())
    # print(pricer.optPayOffAtMaturity())
    print(pricer.price(showAllStates=True))

if __name__ == '__main__':
    demo()
