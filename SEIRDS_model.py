import numpy as np
from topography import nearest_neighbour_topography

class SEIRDS_Model:
    """
    Model represents the topographical SEIR model as vectors of
    floating point numbers. The vectors are Susceptible, Exposed,
    Infected, Resistant and Dead.

    The process is defined by the differential equations, where
    S, E, I, R, D are as defined above and N = S + E + I + R + D (constant):

    dS/dt = rho * R - beta * S * I / N
    dE/dt = beta * S * I / N - sigma * E
    dI/dt = sigma * E - (gamma + digamma) * I
    dR/dt = gamma * I - rho * R
    dD/dt = digamma * I

    The model is made topographical by making each of S, E, I, R, D
    vectors and the topography matrix T is used to multiply the 
    beta term thus:

    dS/dt = rho * R - beta * (S / N) . T * I
    dE/dt = beta * (S / N) . T * I - sigma * E

    Where . T represents matrix multiplication
    """

    def __init__(
            self, 
            populations: np.ndarray,
            beta: float,
            sigma: float,
            gamma: float,
            digamma: float,
            rho: float,
            ):
        """
        Initial state is with all cells susceptible.
        """

        self.beta = beta
        self.sigma = sigma
        self.gamma = gamma
        self.digamma = digamma
        self.rho = rho
            
        shape = populations.shape

        self.s = populations.copy()
        self.e = np.zeros(shape)
        self.i = np.zeros(shape)
        self.r = np.zeros(shape)
        self.d = np.zeros(shape)
        self.n = populations
        self.scale = 1.0 / populations

    def infected(self):
        return self.i

    def __str__(self) -> str:
        NL = "\n"
        return f"s{self.s}{NL}e{self.e}{NL}i{self.i}{NL}r{self.r}{NL}d{self.d}"

    def number_dead(self) -> float:
        return np.sum(self.d)

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
        newly_dead = self.digamma * dt * self.i
        newly_susceptible = self.rho * dt * self.r

        self.s += newly_susceptible - newly_exposed
        self.e += newly_exposed - newly_infected
        self.i += newly_infected - newly_resistant - newly_dead
        self.r += newly_resistant - newly_susceptible
        self.d += newly_dead

def test_one_step_identity_topography():

    populations = np.full((3, 4), 100.0)
    model = SEIRDS_Model(populations, 1.0, 1.0, 1.0, 1.0, 1.0)
    model.infect((0, 0))

    print("model before:")
    print(f"{model}")

    topology = np.identity(populations.size)
    model.timestep(1.0 / 365.0, topology)

    print("model after:")
    print(f"{model}")

def test_365_steps_identity_topography():

    beta = 3.0 * 26.0 # infect three people in the space of two weeks
    sigma = 52.0  # about one week to change from exposed to infected
    gamma = 26.0  # about two weeks infected
    digamma = 0.26  # about 1% of those infected die
    rho = 1.0     # about one year to become susceptible again

    populations = np.full((3, 4), 100.0)
    model = SEIRDS_Model(populations, beta, sigma, gamma, digamma, rho)
    model.infect((0, 0))

    print("model before:")
    print(f"{model}")

    topology = np.identity(populations.size)
    for _ in range(365):
        model.timestep(1.0 / 365.0, topology)

    print("model after:")
    print(f"{model}")

def test_365_steps_nearest_neighbour_topography():

    beta = 3.0 * 26.0 # infect three people in the space of two weeks
    sigma = 52.0  # about one week to change from exposed to infected
    gamma = 26.0  # about two weeks infected
    digamma = 0.26  # about 1% of those infected die
    rho = 1.0     # about one year to become susceptible again

    populations = np.full((3, 4), 100.0)
    model = SEIRDS_Model(populations, beta, sigma, gamma, digamma, rho)
    model.infect((0, 0))

    print("model before:")
    print(f"{model}")

    topology = nearest_neighbour_topography(populations.shape, 1.0, 0.1)
    for _ in range(365):
        model.timestep(1.0 / 365.0, topology)

    print("model after:")
    print(f"{model}")

def test_total_dead_by_beta():

    sigma = 52.0  # about one week to change from exposed to infected
    gamma = 26.0  # about two weeks infected
    digamma = 0.26  # about 1% of those infected die
    rho = 1.0     # about one year to become susceptible again

    populations = np.full((3, 4), 100.0)
    for beta in range(5, 200, 5):
        model = SEIRDS_Model(populations, beta, sigma, gamma, digamma, rho)
        model.infect((0, 0))
        topology = nearest_neighbour_topography(populations.shape, 1.0, 0.1)
        for _ in range(365 * 20):
            model.timestep(1.0 / 365.0, topology)

        print(f"beta={beta} number_dead={model.number_dead()}")

if __name__ == "__main__":
    test_one_step_identity_topography()
    test_365_steps_identity_topography()
    test_365_steps_nearest_neighbour_topography()
    test_total_dead_by_beta()
