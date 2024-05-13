import numpy as np
import random
import numpy.random as rnd

class Bids:
    def __init__(self, n_bids):
        self.n_bids = n_bids

def complex_variation(time, trend_factor=0.01, amplitude=0.2, frequency=2*np.pi/10, threshold=0.5):
  # Linear trend component
  trend = 1 + trend_factor * time

  # Sinusoidal oscillation component
  oscillation = amplitude * np.sin(frequency * time)

  # Combine trend and oscillation, ensuring values between 0 and 1
  combined = np.clip(trend + oscillation, 0, 1)

  # Introduce a step change with a random probability
  if random.random() < 0.1:  # Adjust probability for desired frequency
    if combined > threshold:
      return 0.2  # Change to a lower value after threshold crossed
    else:
      return 0.8  # Change to a higher value before threshold crossed

  return combined

def generate_bids(N, time_variation_factor, current_time):
  
  bids = []
  for _ in range(N):
    # Sample from base distribution (already scaled between 0 and 1)
    base_bid = np.random.uniform(0, 1)

    # Apply time variation
    time_variation = complex_variation(current_time, trend_factor=0.01, amplitude=time_variation_factor, frequency=2*np.pi/10, threshold=0.5)
    modified_bid = base_bid * time_variation

    # Clip the bid to the range (0, 1)
    bid = np.clip(modified_bid, 0, 1)

    bids.append(bid)

  return bids
