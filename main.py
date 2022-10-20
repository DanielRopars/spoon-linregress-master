import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self, xs, ys):

        if xs is None:
            xs = np.zeros(10)

        if ys is None:
            ys = np.zeros(10)

        self.xs = xs
        self.ys = ys
        self.slope = 0
        self.intercept = 0
        self.slope_hist = []
        self.intercept_hist = []

    def loss(self,slope=None,intercept=None):
        # | |I || |_
        if intercept is None:
            intercept = self.intercept

        if slope is None:
            slope=self.slope
        return np.sum(((slope*self.xs+intercept)-self.ys)**2)

    def fit(self, slope, delta=.0001):
        for i in range(1, 1000000):

            gradient_slope = (self.loss(slope=self.slope+delta) - self.loss(slope=self.slope-delta))/(2*delta)
            gradient_intercept = (self.loss(intercept=self.intercept+delta) - self.loss(intercept=self.intercept-delta))/(2*delta)
            min_slope = self.slope
            min_intercept = self.intercept
            min_loss = self.loss(self.slope,self.intercept)

            for gamma in np.logspace(-10,-6,100):
                new_slope = self.slope - gamma*gradient_slope
                new_intercept = self.intercept - gamma*gradient_intercept
                new_loss = self.loss(slope=new_slope,intercept=new_intercept)
                if new_loss < min_loss:
                    min_loss = new_loss
                    min_slope = new_slope
                    min_intercept = new_intercept
                    final_gamma = gamma
            self.slope = min_slope
            self.intercept = min_intercept


            print(f"{min_slope}, {min_intercept}, {min_loss}")


        print(min_intercept)
        print(min_slope)
        print(final_gamma)
        print(min_loss)

    def predict(self, xs, slope):
        return xs*self.slope
        pass


def main():
    adelie_bill_len_mm = np.loadtxt("adelie.csv", delimiter=',', skiprows=1, usecols=0)
    adelie_flipper_len_mm = np.loadtxt("adelie.csv", delimiter=',', skiprows=1, usecols=1)

    lg = LinearRegression(adelie_bill_len_mm, adelie_flipper_len_mm)
    loss_xs = np.linspace(-5,5,100)
    loss_ys = np.array([lg.loss(a) for a in loss_xs])
    plt.plot(loss_xs,loss_ys)
    plt.show()
    lg.fit(5)

if __name__ == '__main__':
    main()