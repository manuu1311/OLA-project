import numpy as np

class PricingEnvironment():
    def __init__(self, conversion_probability, cost, prices):
        self.conversion_probability = conversion_probability
        self.cost = cost
        self.prices = prices

    def round(self, p_t, n_t, day):
        rates = self.conversion_probability[day]
        d_t = np.random.binomial(n_t, rates(p_t))
        r_t = (p_t - self.cost)*(d_t)
        l_t = np.array([(1-(price-self.cost)*rates(price)) for price in self.prices])
        return l_t , r_t , d_t
    

    