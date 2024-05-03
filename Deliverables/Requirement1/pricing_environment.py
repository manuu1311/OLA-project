class PricingEnvironment:
    def __init__(self, distribution, cost):
        self.distribution=distribution
        self.cost = cost

    def round(self, p_t, n_t):
        d_t = self.distribution(p_t,n_t)
        r_t = (p_t - self.cost)*d_t
        return d_t, r_t