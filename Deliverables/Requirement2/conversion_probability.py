import numpy as np

class conversion_probability:
    def __init__(self):
        pass

    def demand_curve(time):
        curve_factor = np.random.uniform(1.5,3)
        trend = 0.05 * np.sin(time * 0.1)
        fluctuation = np.random.uniform(-0.01, 0.01)
        demand = lambda price: min(1, ((curve_factor**(-price/(20)))) + trend + fluctuation)
        return demand






