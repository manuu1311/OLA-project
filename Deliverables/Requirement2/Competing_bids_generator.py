import numpy as np
import random
import numpy.random as rnd

class Bids:
    def __init__(self, n_bids):
        self.n_bids = n_bids

'''
def generate_bids(size, t, T):
        adversarial_bids = np.array([])
        pattern = lambda t: 1 - np.abs(np.sin(5*t/T))
        for i in range(size):
            
            if t < T/3:
                bid = np.random.uniform(0, pattern(t))  
            elif t < (T*2)/3:
                bid = np.random.uniform(0, 1) 
            else:
                bid = np.random.uniform(pattern(t), 1) 
            bid = np.random.uniform(0, pattern(t))
            adversarial_bids = np.append(adversarial_bids, bid)
    
        return adversarial_bids
'''
def complex_variation(time, trend_factor=0.1, amplitude=0.2, frequency=2*np.pi/100, threshold=0.5):
  # Linear trend component
  trend = 1 + trend_factor * time
  # Sinusoidal oscillation component
  oscillation = amplitude * np.sin(frequency * time)
  # Combine trend and oscillation
  combined = np.clip(trend + oscillation, 0, 1)
  # Introduce a step change with a random probability
  if random.random() < 0.1:  
    if combined > threshold:
      return 0.2 
    else:
      return 0.8  
  return combined


def generate_bids(N, time_variation_factor, current_time):
  bids = []
  for _ in range(N):
    base_bid = np.random.uniform(0, 1)
    time_variation = complex_variation(current_time, trend_factor=0.1, amplitude=time_variation_factor, frequency=2*np.pi/5, threshold=0.5)
    modified_bid = base_bid * time_variation
    #if np.random.choice([True, False]) and modified_bid > 0.8:
      #modified_bid -= 0.2
    bid = np.clip(modified_bid, 0, 1)
    bids.append(bid)
  return np.array(bids)