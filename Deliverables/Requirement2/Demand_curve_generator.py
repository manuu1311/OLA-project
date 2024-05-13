import numpy as np

class Demand_curve_generator():
    def __init__(self):
        pass

    # this method returns the probability of sale for a general product in function of product's price.
    # The output curve varies based on the steep parameter randomly generated when the function is invoked. 
    def generate_demand(min_price, max_price , curve_steep):

        conversion_probability = lambda price: np.clip(np.exp(-((np.clip(price, 0, 1) - min_price) / (max_price - min_price)) * curve_steep), 0, 1)

        return conversion_probability
                
