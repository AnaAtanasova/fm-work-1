import numpy as np
from scipy.optimize import newton


class Bond:
    def __init__(self, nominal, delta, num_payments, annual_rate, observed_price):
        self.nominal = nominal
        self.delta = delta
        self.num_payments = num_payments
        self.annual_rate = annual_rate
        self.observed_price = observed_price
        # use an underscore to indicate that this attribute should not be accessed directly
        self._calibrated_rate = None

    def __str__(self):
        return f"Bond(nominal={self.nominal}, delta={self.delta}, num_payments={self.num_payments}, annual_rate={self.annual_rate}, observed_price={self.observed_price})"

    def __repr__(self):
        return str(self)

    def f(self, x):
        """
        Calibration function for the Bond object. Computes the present value of the bond using the given parameters
        and an input rate x, and returns the error between the observed price and the computed present value.
        """
        payment = self.nominal * self.annual_rate / self.delta
        present_value = sum(payment / ((1 + x)**(i*self.delta))
                            for i in range(1, self.num_payments+1))

        present_value += (self.nominal + payment) / \
            ((1 + x)**(self.num_payments*self.delta))
        error = self.observed_price - present_value
        return error

    def calibrate(self):
        """
        Calibrates the Bond object by finding the root of the calibration function using scipy.optimize.newton().
        """
        x0 = 0.05  # initial guess for the root
        tol = 1e-6  # tolerance for the root-finding algorithm
        maxiter = 100  # maximum number of iterations for the root-finding algorithm
        # use scipy.optimize.newton() to find the root
        x = newton(self.f, x0, fprime=None, tol=tol, maxiter=maxiter)
        self._calibrated_rate = x  # set the value of the calibrated_rate property
        return x

    @property
    def calibrated_rate(self):
        """
        Property getter method that returns the calibrated rate.
        """
        if self._calibrated_rate is None:
            self.calibrate()
        return self._calibrated_rate


bonds = [
    Bond(100, 0.25, 4, 0.03, 111),
    Bond(100, 0.5, 5, 0.03, 111),
    Bond(100, 1, 6, 0.02, 102),
    Bond(100, 0.5, 10, 0.03, 123)
]

for bond in bonds:
    print(f"{bond} -> calibrated rate: {bond.calibrated_rate:.4f}")
