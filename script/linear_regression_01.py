#!/usr/bin/env python3
#coding:utf-8

"""
確率的な線型回帰。
基底関数の線型結合でサンプルによくフィットする関数を表現する。
最小化するのは各サンプルの誤差の二乗和。
線型結合の係数には事前分布として平均0の正規分布を仮定する。
結果は上記係数の事後分布として得る。この事後分布も正規分布となる。
係数の事後分布が正規分布として得られれば、新しい入力xに対する出力yも正規分布として得られる。
"""

import numpy as np
import matplotlib.pyplot as plt

def generate_samples(num_of_samples, x_range_min, x_range_max, noise_sigma):

    samples_x = np.zeros(num_of_samples)
    samples_y = np.zeros(num_of_samples)
    
    noises = np.random.normal(0, noise_sigma, num_of_samples)
    
    samples_x = np.random.rand(num_of_samples) * (x_range_max - x_range_min) - x_range_min
    samples_y = np.sin(samples_x) + noises

    return (samples_x, samples_y)

def estimate(samples_x, samples_y, range_x_min, range_x_max, prior_distribution_cov, stddev_of_samples):

    num_of_bases = prior_distribution_cov.shape[0] - 1
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

    parameters_cov = np.linalg.inv(phi.T @ phi / stddev_of_samples**2 + np.linalg.inv(prior_distribution_cov))
    
    return (bases, parameters_cov @ phi.T @ samples_y / stddev_of_samples**2, parameters_cov)

def plot(bases, parameters_mean, parameters_cov, stddev_of_samples, range_x_min, range_x_max, samples_x, samples_y):

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(samples_x, samples_y, marker='.')
    N = 1000
    tmp_x = np.linspace(range_x_min, range_x_max, N)
    tmp_y = np.zeros(N)
    tmp_phi = np.zeros((len(parameters_mean), N))
    tmp_uncertainty = np.zeros(N)
    for i, w in enumerate(parameters_mean):
        tmp_phi[i, :] = bases[i](tmp_x)
        tmp_y += w * tmp_phi[i, :]
    tmp1 = parameters_cov @ tmp_phi
    for i in range(N):
        tmp_uncertainty[i] = np.dot(tmp_phi[:, i], tmp1[:, i]) + stddev_of_samples ** 2
    ax.plot(tmp_x, tmp_y + 3 * tmp_uncertainty, color="gray")
    ax.plot(tmp_x, tmp_y - 3 * tmp_uncertainty, color="gray")
    ax.plot(tmp_x, tmp_y, color="black")
    ax.grid()
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.show()

if __name__ == '__main__':

    range_x_min = 0
    range_x_max = np.pi

    samples_x, samples_y = generate_samples(10, range_x_min, range_x_max, 0.1)

    prior_distribution_cov = np.eye(6)
    stddev_of_samples = 0.1
    bases, parameters_mean, parameters_cov = estimate(samples_x, samples_y, range_x_min, range_x_max, prior_distribution_cov, stddev_of_samples)

    plot(bases, parameters_mean, parameters_cov, stddev_of_samples, range_x_min, range_x_max, samples_x, samples_y)




