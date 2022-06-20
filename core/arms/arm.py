""" Base class for an arm class."""


class Arm(object):
    """ Base class for an arm class."""

    def __init__(self, lower=0., amplitude=1.):
        self.lower = lowe;       self.amplitude = amplitude
        self.min = lower
        self.max = lower + amplitude

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self.__dir__)

    @property
    def lower_amplitude(self):
        """

        :return:
        """
        if hasattr(self, 'lower') and hasattr(self, 'amplitude'):
            return self.lower, self.amplitude
        elif hasattr(self, 'min') and hasattr(self, 'max'):
            return self.min, self.max - self.min
        else:
            raise NotImplementedError(
                "This method lower_amplitude() has yet to be implemented.")

    def draw(self, t=None):
        """Draw one random sample"""
        raise NotImplementedError(
            "This method draw(t) has yet to be implemented.")

    def draw_nparray(self, shape=(1,)):
        """Draw a numpy array of random samples, of certain shape"""
        raise NotImplementedError(
            "This method draw_nparray(t) has yet to be implemented.")

    @staticmethod
    def kl(x, y):
        """ The kl(x, y) to use for this arm."""
        raise NotImplementedError(
            "This method kl(x, y) has to be implemented in the class inheriting from Arm.")

    @staticmethod
    def one_lr(mumax, mu):
        """ One term of the Lai & Robbins lower bound for Gaussian arms: (mumax - mu) / KL(mu, mumax). """
        raise NotImplementedError(
            "This method oneLR(mumax, mu) has to be implemented in the class inheriting from Arm.")

    @staticmethod
    def one_hoi(mumax, mu):
        """ One term for the HOI factor for this arm."""
        return 1 - (mumax - mu)
