import numpy as np

class Environment:
    def __init__(self):
        self.day = 0
        self.bids = []
        self.buy_probability = []

    def generate_bids(self):
        # Generate bids using a non-stationary distribution
        mean = np.sin(self.day / 10) * 10  # Mean changes over time
        bids = np.random.normal(mean, 1, size=10)  # 10 agents
        self.bids = bids

    def generate_buy_probability(self):
        # Generate buy probability function using a non-stationary process
        mean = np.cos(self.day / 10)  # Mean changes over time
        self.buy_probability = np.clip(np.random.normal(mean, 0.1, size=100), 0, 1)  # 100 price levels

    def step(self):
        self.generate_bids()
        self.generate_buy_probability()
        self.day += 1