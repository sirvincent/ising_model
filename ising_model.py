# Simulation of the ising model by using the metroppolis algorithm
# TODO: clear from code/name?
import numpy as np

SPINS_1D = 5
SPINS_TOTAL = SPINS_1D * SPINS_1D


class Lattice:
    """
    The lattice containing up and down spins of 1/2 magnetisatie
    """
    def __init__(self, initialization_method):
        # TODO: boolean lattice initialization might save even more memory
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








if __name__ == '__main__':
    lattice = Lattice('down')
    print(lattice.lattice)
