import numpy as np
import math
import matplotlib.pyplot as plt

class BlackKarasinski:
    def __init__(self, T, N, r_0, alpha, beta, sigma):
        self._T = T
        self._N = N
        self._delta = float(T) / float(N)
        self._alpha = alpha
        self._beta = beta
        self._sigma = sigma

        assert r_0 > .0, 'r_0 must be positive!'
        self._r_0 = r_0
        self._delta = float(T) / float(N)

    def simulate(self, nb_simus, plot_paths=False, nb_paths=5):
        ln_paths = np.zeros((nb_simus, self._N))
        paths = np.zeros((nb_simus, self._N))
        dW_t = np.random.normal(size =(nb_simus, self._N)) * math.sqrt(self._delta)

        for simu in range(0, nb_simus):
            ln_paths[simu, 0] = math.log(self._r_0)

        for time in range(1, self._N):
            for simu in range(0, nb_simus):
                ln_paths[simu, time] = ln_paths[simu, time-1] + (self._alpha - self._beta * ln_paths[simu, time-1]) * self._delta + self._sigma * dW_t[simu, time]

        for time in range(1, self._N):
            for simu in range(0, nb_simus):
                paths[simu, time] = math.exp(ln_paths[simu, time])

        if plot_paths:
            times = [t * self._delta for t in range(0, self._N)]
            y_max = .0

            for i, path in enumerate(paths): 
                if i <= nb_paths:
                    plt.plot(times, paths[i], linewidth=.4, color='blue')
                    max_i = max(paths[i])
                    if max_i > y_max:
                        y_max = max_i

            plt.xlim(.0, self._T)
            plt.ylim(.0, y_max)

            plt.show()

        return paths
