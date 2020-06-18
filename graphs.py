import matplotlib.pyplot as plt
import numpy as np
from topography import nearest_neighbour_topography, exponential_topography, stratified_topography
from SEIR_model import SEIR_Model
from SEIRDS_model import SEIRDS_Model
from math import exp, log

def evolve(model, topography, has_d, tau):

    STEPS = 365
    timesteps = np.arange(0, STEPS, 1)
    ONE_DAY = 1.0 / 365.0

    S = np.zeros(STEPS)
    E = np.zeros(STEPS)
    I = np.zeros(STEPS)
    R = np.zeros(STEPS)
    if has_d:
        D = np.zeros(STEPS)
    r_factor = np.zeros(STEPS)
    N = np.sum(model.s)

    for i, _ in enumerate(timesteps):
        model.timestep(ONE_DAY, topography)

        S[i] = np.sum(model.s)
        E[i] = np.sum(model.e)
        I[i] = np.sum(model.i)
        R[i] = np.sum(model.r)
        if has_d:
            D[i] = np.sum(model.d)
        
        if i > 0:
            effective_beta = (S[i - 1] - S[i]) * N / (S[i - 1] * I[i - 1] * ONE_DAY)
            r_factor[i - 1] = effective_beta * tau

    # fudge the last points for prettiness
    r_factor[-1] = r_factor[-2]

    fig, ax = plt.subplots()
    ax.plot(timesteps, r_factor, 'b', label="R-factor")

    ax.set(xlabel='time (days)', ylabel='factors',
        title='SEIR model r factor')
    ax.grid()
    plt.legend()
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(timesteps, S, 'b', label='Susceptible')
    ax.plot(timesteps, E, 'g', label='Exposed')
    ax.plot(timesteps, I, 'y', label='Infected')
    ax.plot(timesteps, R, 'c', label='Resistant')
    if has_d:
        ax.plot(timesteps, D, 'k', label='Dead')

    name = "SEIRD" if has_d else "SEIR"

    ax.set(xlabel='time (days)', ylabel='population',
        title=f'{name} model')
    ax.grid()
    plt.legend()
    fig.savefig(f"{name}.png")
    plt.show()

def evolve_SEIR():
    beta = 3.0 * 26.0 # infect three people in the space of two weeks
    sigma = 52.0  # about one week to change from exposed to infected
    gamma = 26.0  # about two weeks infected

    populations = np.full((100, 100), 100.0)
    model = SEIR_Model(populations, beta, sigma, gamma)
    model.infect((0, 0))

    topography = nearest_neighbour_topography(populations.shape, 1.0, 0.1)
    evolve(model, topography, False, 1.0 / gamma)

def evolve_SEIR_stratified():
    beta = 3.0 * 26.0 # infect three people in the space of two weeks
    sigma = 52.0  # about one week to change from exposed to infected
    gamma = 26.0  # about two weeks infected

    pop_multiplier = 3000.0
    decay = 0.05
    populations = np.exp(np.fromfunction(lambda i, j: -decay * j, (1, 100))) * pop_multiplier
    # populations = np.full((1, 100), 100.0)
    print(f"{populations}")
    model = SEIR_Model(populations, beta, sigma, gamma)
    model.infect((0, 0))

    #topography = stratified_topography(populations.shape, 10.0, 0.6, 0.6)
    topography = stratified_topography(populations.shape, 1.0, 0.6, 0.6)
    evolve(model, topography, False, 1.0 / gamma)

def evolve_SEIRDS():

    beta = 3.0 * 26.0 # infect three people in the space of two weeks
    sigma = 52.0  # about one week to change from exposed to infected
    gamma = 26.0  # about two weeks infected
    digamma = 0.26  # about 1% of those infected die
    rho = 1.0     # about one year to become susceptible again

    populations = np.full((30, 30), 100.0)
    model = SEIRDS_Model(populations, beta, sigma, gamma, digamma, rho)
    model.infect((0, 0))

    topography = exponential_topography(populations.shape, 1.0, 1.5)
    #topography = nearest_neighbour_topography(populations.shape, 1.0, 0.1)
    evolve(model, topography, True, 1.0 / gamma)


def evolve_SEIRDS_stratified():
    beta = 3.0 * 26.0 # infect three people in the space of two weeks
    sigma = 52.0  # about one week to change from exposed to infected
    gamma = 26.0  # about two weeks infected
    digamma = 0.26  # about 1% of those infected die
    rho = 1.0     # about one year to become susceptible again

    pop_multiplier = 3000.0
    decay = 0.05
    populations = np.exp(np.fromfunction(lambda i, j: -decay * j, (1, 100))) * pop_multiplier
    # populations = np.full((1, 100), 100.0)
    print(f"{populations}")
    model = SEIRDS_Model(populations, beta, sigma, gamma, digamma, rho)
    model.infect((0, 0))

    #topography = stratified_topography(populations.shape, 10.0, 0.6, 0.6)
    topography = stratified_topography(populations.shape, 1.0, 0.6, 0.6)
    evolve(model, topography, True, 1.0 / gamma)

if __name__ == '__main__':
    evolve_SEIR_stratified()