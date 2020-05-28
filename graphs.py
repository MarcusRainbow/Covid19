import matplotlib.pyplot as plt
import numpy as np
from topography import nearest_neighbour_topography, exponential_topography
from SEIR_model import SEIR_Model
from SEIRDS_model import SEIRDS_Model

def evolve(model, topography):

    STEPS = 365 * 10
    timesteps = np.arange(0, STEPS, 1)
    ONE_DAY = 1.0 / 365.0

    S = np.zeros(STEPS)
    E = np.zeros(STEPS)
    I = np.zeros(STEPS)
    R = np.zeros(STEPS)
    D = np.zeros(STEPS)

    for i, _ in enumerate(timesteps):
        model.timestep(ONE_DAY, topography)

        S[i] = np.sum(model.s)
        E[i] = np.sum(model.e)
        I[i] = np.sum(model.i)
        R[i] = np.sum(model.r)
        D[i] = np.sum(model.d)

    fig, ax = plt.subplots()
    ax.plot(timesteps, S, 'b', label='Susceptible')
    ax.plot(timesteps, E, 'g', label='Exposed')
    ax.plot(timesteps, I, 'y', label='Infected')
    ax.plot(timesteps, R, 'c', label='Resistant')
    ax.plot(timesteps, D, 'k', label='Dead')

    ax.set(xlabel='time (days)', ylabel='population',
        title='SEIRD model')
    ax.grid()

    plt.legend()

    fig.savefig("SEIRD.png")
    plt.show()

def evolve_SEIR():
    beta = 3.0 * 26.0 # infect three people in the space of two weeks
    sigma = 52.0  # about one week to change from exposed to infected
    gamma = 26.0  # about two weeks infected

    populations = np.full((100, 100), 100.0)
    model = SEIR_Model(populations, beta, sigma, gamma)
    model.infect((0, 0))

    topography = nearest_neighbour_topography(populations.shape, 1.0, 0.1)
    evolve(model, topography)

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
    evolve(model, topography)

if __name__ == '__main__':
    evolve_SEIRDS()