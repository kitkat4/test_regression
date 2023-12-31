#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

def generate_samples(num_of_samples, x_range_min, x_range_max, noise_sigma):

    samples_x = np.zeros(num_of_samples)
    samples_y = np.zeros(num_of_samples)
    
    noises = np.random.normal(0, noise_sigma, num_of_samples)
    
    samples_x = np.random.rand(num_of_samples) * (x_range_max - x_range_min) - x_range_min
    samples_y = np.sin(samples_x) + noises

    return (samples_x, samples_y)

def estimate(samples_x, samples_y, range_x_min, range_x_max):

    num_of_bases = 5
    bases = []

    basis_sigma = 0.5

    bases.append(lambda x : 1)

    for i in range(num_of_bases):
        bases.append(lambda x, i=i : np.exp(-(x - (range_x_min + i * (range_x_max - range_x_min) / (num_of_bases - 1)))**2/(2 * basis_sigma ** 2)) / np.sqrt(2 * np.pi * basis_sigma**2))
        # bases.append(lambda x, i=i : x**(i+1))

    phi = np.zeros((len(samples_x), len(bases)))

    for i in range(phi.shape[0]):
        for j in range(phi.shape[1]):
            phi[i, j] = bases[j](samples_x[i])


    return (bases, np.linalg.inv(phi.T @ phi) @ phi.T @ samples_y)

def plot(bases, parameters, range_x_min, range_x_max, samples_x, samples_y):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(samples_x, samples_y, marker='.')
    tmp_x = np.linspace(range_x_min, range_x_max, 1000)
    tmp_y = np.zeros(1000)
    for i, w in enumerate(parameters):
        tmp_y += w * bases[i](tmp_x)
    ax.plot(tmp_x, tmp_y, color="black")
    ax.grid()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()

if __name__ == '__main__':

    range_x_min = 0
    range_x_max = np.pi

    samples_x, samples_y = generate_samples(1000, range_x_min, range_x_max, 0.1)

    bases, parameters = estimate(samples_x, samples_y, range_x_min, range_x_max)

    plot(bases, parameters, range_x_min, range_x_max, samples_x, samples_y)




