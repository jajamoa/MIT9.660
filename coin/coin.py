"""
Author: Chance Jiajie Li
Email: jiajie@media.mit.edu
"""

import numpy as np
import matplotlib.pyplot as plt

coin_sequences = {
    "coin1": "HHTHT",
    "coin2": "THTTT",
    "coin3": "HHHHH",
    "coin4": "THTTHTHTHT",
    "coin5": "HHTHHHHHTH",
    "coin6": "TTTTTTTTTT",
    "coin7": "THTTHTTHHTHTHTTHTHTTTHTTHT",
    "coin8": "HHTHHHHTHHHTHHHTHHHHHTHHHH",
    "coin9": "HHHHHHHHHHHHHHHHHHHHHHHHHH"
}

human_datasets = {
    "First Cover Story + Myself": np.array([3, 5, 7, 2, 6, 7, 2, 6, 7]),
    "First Cover Story + Friend": np.array([3, 4, 5, 4, 6, 7, 3, 6, 7]),
    "Second Cover Story + Myself": np.array([4, 5, 5, 2, 5, 6, 1, 5, 7]),
    "Second Cover Story + Friend": np.array([3, 3, 5, 2, 3, 6, 1, 6, 6]),
}

# Initialize a dictionary to hold the count of 'H' and 'T' for each coin sequence
coin_statistics = {}

# Loop over each coin sequence and count the occurrences of 'H' and 'T'
for coin, sequence in coin_sequences.items():
    H_count = sequence.count('H')
    T_count = sequence.count('T')
    coin_statistics[coin] = {"H": H_count, "T": T_count}

def logistic_transform(x, a=2, b=0):
    return 1 / (1 + np.exp(-a * x + b))

def log_posterior_odds_ratio(p_d_h1, p_d_h2, p_h1 = 0.5):
    p_h2 = 1 - p_h1
    return np.log(p_d_h1 / p_d_h2) + np.log(p_h1 / p_h2)

def probability_sequence_given_H1(sequence_length):
    return 1 / (2 ** sequence_length)

def probability_sequence_given_H2(H, T):
    # Define 100 discrete theta values between 0 and 1 to approximate the integral
    theta_values = np.linspace(0, 1, 100)
    probabilities = []
    
    for theta in theta_values:
        # For each theta value, calculate P(D | theta)P(theta | H2)
        # P(theta | H2) = 1/100 is the probability for each theta value
        prob = (theta ** H) * ((1 - theta) ** T) * (1 / 100)
        probabilities.append(prob)
    
    # Sum up the probabilities to get P(D | H2)
    return sum(probabilities)

scaled_predictions = []

for coin, stats in coin_statistics.items():
    print(f"{coin}: Heads={stats['H']}, Tails={stats['T']}")
    p_d_h1 = probability_sequence_given_H1(stats['H']+stats['T'])
    p_d_h2 = probability_sequence_given_H2(stats['H'], stats['T'])
    predictions = logistic_transform(log_posterior_odds_ratio(p_d_h1, p_d_h2))
    # Scale predictions to the range [1, 7]
    scaled_prediction = 6 * predictions + 1
    scaled_predictions.append(scaled_prediction)
    print(scaled_prediction, np.round(scaled_prediction))

# Process each human dataset
for dataset_name, human_data in human_datasets.items():
    # Plotting the scaled model predictions against the human data
    plt.scatter(scaled_predictions, 8 - human_data)
    plt.xlabel('Scaled Model Predictions')
    plt.ylabel('Human Data')
    plt.title(f'{dataset_name}: Scaled Model Predictions vs Human Data')
    # Drawing the line of perfect fit (baseline) y=x
    plt.plot([1, 7], [1, 7], 'k--', label='Baseline (y=x)')
    # Setting the axis range to [1, 7] for both x and y axis
    plt.axis([1, 7, 1, 7])
    plt.show()
    
    # Calculating and reporting the correlation coefficient
    correlation_matrix = np.corrcoef(scaled_predictions, 8 - human_data)
    correlation_coefficient = correlation_matrix[0, 1]
    print(f'{dataset_name}: Correlation Coefficient: {correlation_coefficient}')