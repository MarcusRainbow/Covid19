import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from topography import nearest_neighbour_topography
from SEIR_model import SEIR_Model
from SEIRDS_model import SEIRDS_Model

def evolve(model, topography):

    fig = plt.figure()
    plt.axis("off")
    frames = []

    ONE_DAY = 1.0 / 365.0

    for _ in range(365 * 5):
        model.timestep(ONE_DAY, topography)

        # animation
        frames.append([plt.imshow(model.infected())])

    animation.ArtistAnimation(fig, frames, interval=25, blit=True,
                                repeat_delay=1000)
    #a.save("SEIR.html")

    plt.show()

def evolve_SEIR():
    beta = 3.0 * 26.0 # infect three people in the space of two weeks
    sigma = 52.0  # about one week to change from exposed to infected
    gamma = 26.0  # about two weeks infected

    populations = np.full((30, 30), 100.0)
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

    populations = np.full((100, 100), 100.0)
    model = SEIRDS_Model(populations, beta, sigma, gamma, digamma, rho)
    model.infect((0, 0))

    topography = nearest_neighbour_topography(populations.shape, 1.0, 0.1)
    evolve(model, topography)

if __name__ == '__main__':
    evolve_SEIRDS()