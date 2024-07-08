import numpy as np
import random
import numpy.random as rnd

class Bids:
    def __init__(self, n_bids):
        self.n_bids = n_bids

def complex_variation(time, trend_factor=0.1, amplitude=0.2, frequency=2*np.pi/100, threshold=0.5):
  trend = 1 + trend_factor * time
  oscillation = amplitude * np.sin(frequency * time)
  combined = np.clip(trend + oscillation, 0, 1)
  if random.random() < 0.2:  
    if combined > threshold:
      return 0.2 
    else:
      return 0.8  
  if random.random() < 0.5:  
     combined -= 0.2
  return combined


def generate_bids(N, time_variation_factor, current_time):
  bids = []
  for _ in range(N):
    base_bid = np.random.uniform(0, 1)
    time_variation = complex_variation(current_time, trend_factor=0.1, amplitude=time_variation_factor, frequency=2*np.pi/5, threshold=0.5)
    modified_bid = base_bid * time_variation
    bid = np.clip(modified_bid, 0, 1)
    bids.append(bid)
  return np.array(bids)