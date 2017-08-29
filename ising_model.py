# Simulation of the 2D ising model by using the metropolis algorithm
import numpy as np
import matplotlib.pyplot as plt

SPINS_1D = 5 
SPINS_TOTAL = SPINS_1D * SPINS_1D
J = 1  # Interaction energy
k = 1  # Boltzmann constant
# We neglect the external magnetic field B in this ising model simulation
MEASUREMENTS = 1000


class Lattice:
    """
    The lattice containing up and down spins of 1/2 magnetisatie
    """
    def __init__(self, initialization_method, temperature):
        if initialization_method == 'down':
            init_lattice = np.zeros((SPINS_1D, SPINS_1D), dtype=np.int8)
            init_lattice[init_lattice == 0] = -1
            self.lattice = init_lattice
        elif initialization_method == 'up':
            self.lattice = np.ones((SPINS_1D, SPINS_1D), dtype=np.int8)
        elif initialization_method == 'random':
            init_lattice = np.random.randint(0, 2, size=(SPINS_1D, SPINS_1D),
                    dtype = np.int8) 
            init_lattice[init_lattice == 0] = -1
            self.lattice = init_lattice
        else:
            sys.exit('no valid name for initialization lattice; use:'
                     'up/down/random')
        self.determine_energy()
        self.determine_magnetization()
        # Acceptance ratio is equal to 1 if new state is lower or equal in
        # energy otherwise it is equal to the underneath exponential
        # The energy difference is rewritten by plugging
        # in the energy update rule. 
        # Exponential is initialized here with all possible energy update
        # values because recalculating everytime is expensive.
        self.acceptance_ratio = np.exp(-2 * 1/temperature *\
                                       J * np.array([2, 4]))
        self.thermalized = False


    def determine_energy(self):
        # Using periodic boundary conditions 
        self.energy = 0
        for itSpinRow in range(len(self.lattice)):
            for itSpinCol in range(len(self.lattice)):
                center_spin = self.lattice[itSpinRow, itSpinCol]
                nearest_neighbour_spins = self.lattice[(itSpinRow+1) % SPINS_1D, itSpinCol]\
                    + self.lattice[itSpinRow, (itSpinCol+1) % SPINS_1D]\
                    + self.lattice[(itSpinRow + SPINS_1D -1) % SPINS_1D,
                              itSpinCol]\
                    + self.lattice[itSpinRow, (itSpinCol + SPINS_1D - 1) %
                              SPINS_1D]
                self.energy += -nearest_neighbour_spins * center_spin

    def determine_magnetization(self):
        self.magnetization = np.sum(self.lattice)
    
    def single_sweep_metropolis_algorithm(self):
        """
        Single sweep i.e. evolution of the lattice with the metropolis
        algorithm
        """
        for itSpin in range(SPINS_TOTAL):
            random_row = np.random.randint(0, SPINS_1D)
            random_column = np.random.randint(0, SPINS_1D)
            spin = self.lattice[random_row, random_column]
            nearest_neighbour_spins = self.lattice[(random_row+1) % SPINS_1D, random_column]\
                    + self.lattice[random_row, (random_column+1) % SPINS_1D]\
                    + self.lattice[(random_row + SPINS_1D - 1) % SPINS_1D,
                              random_column]\
                    + self.lattice[random_row, (random_column + SPINS_1D - 1) %
                              SPINS_1D]
            energy_difference_with_new_state = 2 * J * spin * nearest_neighbour_spins
            if energy_difference_with_new_state <= 0:
                self.lattice[random_row, random_column] = -spin
                self.energy += energy_difference_with_new_state
                self.magnetization += 2 * self.lattice[random_row, random_column] 
            elif energy_difference_with_new_state > 0:
                if self.acceptance_ratio[int(spin * nearest_neighbour_spins / 2 - 1)]\
                   > np.random.rand():
                    self.lattice[random_row, random_column] = -spin
                    self.energy += energy_difference_with_new_state
                    self.magnetization += 2 * self.lattice[random_row, random_column] 

    def thermalization(self, iteration_until_thermalization):
        for it in range(iteration_until_thermalization):
            self.single_sweep_metropolis_algorithm()
        self.thermalized = True


if __name__ == '__main__':
    temperatureList = np.arange(0.2, 5.2, 0.2)
    # Before measurements are made allow the system to evolve into thermal
    # equilibrium (thermalization)
    # TODO: around the critical temperature (here, T=2.269), increase
    #       thermalization time due to critical slowing down
    thermalizationList = np.repeat(200, len(temperatureList)) 

    magnetizationMeasurements = np.empty((len(temperatureList), MEASUREMENTS))
    energyMeasurements =  np.empty((len(temperatureList), MEASUREMENTS))
    for element, temperature in enumerate(temperatureList):
        lattice = Lattice('up', temperature)
        print('temperature: ', temperature)
        if lattice.thermalized == False:
            lattice.thermalization(thermalizationList[element])
        if lattice.thermalized == True:
            for it in range(MEASUREMENTS):
                magnetizationMeasurements[element, it] = lattice.magnetization
                energyMeasurements[element, it] = lattice.energy
                lattice.single_sweep_metropolis_algorithm()
    
    # Due to that all spins up and down carry the same energy, fluctuations
    # will sometimes kick after a sweep all spins to up or down. Therefore we
    # take the absolute value
    meanMagnetization = np.mean(np.absolute(magnetizationMeasurements),
                                axis=1) / SPINS_TOTAL 

    plt.plot(temperatureList, meanMagnetization)
    plt.xlim([0, 5])
    plt.show()
