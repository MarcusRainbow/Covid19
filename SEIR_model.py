import numpy as np
from topography import nearest_neighbour_topography

class SEIR_Model:
    """
    Model represents the topographical SEIR model as vectors of
    floating point numbers. The vectors are Susceptible, Exposed,
    Infected and Resistant (recovered, quarantined or dead).
    
    The process is defined by the differential equations, where
    S, E, I, R are as defined above and N = S + E + I + R (constant):

    dS/dt = -beta * S * I / N
    dE/dt = beta * S * I / N - sigma * E
    dI/dt = sigma * E - gamma * I
    dR/dt = gamma * I 

    The model is made topographical by making each of S, E, I, R
    vectors and the topography matrix T is used to multiply the 
    beta term thus:

    dS/dt = -beta * (S / N) . T * I
    dE/dt = beta * (S / N) . T * I - sigma * E

    Where . T represents matrix multiplication
    """

    def __init__(
            self, 
            populations: np.ndarray,
            beta: float,
            sigma: float,
            gamma: float):
        """
        Initial state is with all cells susceptible.
        """

        self.beta = beta
        self.sigma = sigma
        self.gamma = gamma
            
        shape = populations.shape

        self.s = populations.copy()
        self.e = np.zeros(shape)
        self.i = np.zeros(shape)
        self.r = np.zeros(shape)
        self.n = populations
        self.scale = 1.0 / populations

    def __str__(self) -> str:
        NL = "\n"
        return f"s{self.s}{NL}e{self.e}{NL}i{self.i}{NL}r{self.r}"

    def infect(self, cell: (int, int), infection: float = 1.0):
        """
        Infect just one cell by converting a susceptible individual to infected
        """
        self.s[cell] -= infection
        self.i[cell] += infection

    def timestep(
            self,
            dt: float,
            topography: np.ndarray):
        """
        Time evolve each cell of the model by one timestep. The size of the timestep
        is dt. The rate of exposure is beta. The rate of infection is
        sigma. The rate of recovery/quarantine/death is gamma.
        """

        size = self.n.size
        assert topography.shape == (size, size)

        # This is the standard SEIR model, except that we also allow neighbouring cells
        # to expose susceptible individuals according to the given topology matrix.

        infectious = self.beta * dt * self.scale * self.i
        infectious.shape = (1, size)
        exposure = infectious.dot(topography)
        exposure.shape = self.s.shape
        newly_exposed = exposure * self.s
        newly_infected = self.sigma * dt * self.e
        newly_resistant = self.gamma * dt * self.i

        self.s -= newly_exposed
        self.e += newly_exposed - newly_infected
        self.i += newly_infected - newly_resistant
        self.r += newly_resistant

def test_one_step_identity_topography():

    populations = np.full((3, 4), 100.0)
    model = SEIR_Model(populations, 1.0, 1.0, 1.0)
    model.infect((0, 0))

    print("model before:")
    print(f"{model}")

    topology = np.identity(populations.size)
    model.timestep(1.0 / 365.0, topology)

    print("model after:")
    print(f"{model}")

def test_365_steps_identity_topography():

    populations = np.full((3, 4), 100.0)
    model = SEIR_Model(populations, 20.0, 10.0, 10.0)
    model.infect((0, 0))

    print("model before:")
    print(f"{model}")

    topology = np.identity(populations.size)
    for _ in range(365):
        model.timestep(1.0 / 365.0, topology)

    print("model after:")
    print(f"{model}")

def test_365_steps_nearest_neighbour_topography():

    populations = np.full((3, 4), 100.0)
    model = SEIR_Model(populations, 20.0, 10.0, 10.0)
    model.infect((0, 0))

    print("model before:")
    print(f"{model}")

    topology = nearest_neighbour_topography(populations.shape, 1.0, 0.1)
    for _ in range(365):
        model.timestep(1.0 / 365.0, topology)

    print("model after:")
    print(f"{model}")

if __name__ == "__main__":
    test_one_step_identity_topography()
    test_365_steps_identity_topography()
    test_365_steps_nearest_neighbour_topography()



