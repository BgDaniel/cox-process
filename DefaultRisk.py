import random as rnd
import numpy as np
import matplotlib.pyplot as plt
import math

from stoch_processes.StochProcesses import BlackKarasinski

class BernoulliExperiment:
    def __init__(self, p):
        self._p = p
    
    def result(self):
        r = rnd.random()

        if r <= self._p:
            return 1.0
        else:
            return 0.0

class LocalPoissonProcess:
    def __init__(self, intensity, T, N):
        self._intensity = intensity
        self._T = T
        self._N = N
        self._delta = float(T) / float(N)

    def simulate(self, nb_simus, plot_survival_prob=False, plot_paths=False, nb_paths=5, plot_expectation=False):
        paths = np.zeros((nb_simus, self._N))
        times = [t * self._delta for t in range(0, self._N)]
        p_t = np.zeros((nb_simus, self._N))
        bernoullis = np.ndarray(shape=(nb_simus, self._N), dtype=BernoulliExperiment)
        
        for time in range(1, self._N):
            for simu in range(0, nb_simus):
                p_t[simu,time] = self._T * self._intensity[simu, time]  / float(self._N)
                bernoullis[simu,time] = BernoulliExperiment(p_t[simu,time])

        survival_prob = np.zeros(self._N)
        running_defaults = 0

        for time in range(1, self._N):
            for simu in range(0, nb_simus):
                paths[simu,time] = paths[simu,time-1] + bernoullis[simu, time].result()

                if paths[simu,time] < 1.0:
                    survival_prob[time] += 1.0 / float(nb_simus)

        if plot_survival_prob:
            plt.xlim(.0, self._T)
            plt.ylim(.0, 1.1)
            plt.plot(times, survival_prob, linewidth=.4, color='blue')

            integral_expectation = np.zeros(self._N)

            for time in range(1, self._N):
                expectation_t = .0
                for simu in range(0, nb_simus):
                    expectation_t += self._intensity[simu,time] / float(nb_simus)
                
                integral_expectation[time] = integral_expectation[time-1] + expectation_t * self._delta

            plt.plot(times, [math.exp(- integral_expectation[t]) for t in range(0, self._N)], linewidth=.4, color='red')
            plt.show()

        if plot_paths:
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

        if plot_expectation:
            integral_expectation = np.zeros(self._N)

            for time in range(1, self._N):
                expectation_t = .0
                for simu in range(0, nb_simus):
                    expectation_t += self._intensity[simu,time] / float(nb_simus)
                
                integral_expectation[time] = integral_expectation[time-1] + expectation_t * self._delta

            expectation = np.zeros(self._N)

            for time in range(0, self._N):
                for simu in range(0, nb_simus):
                    expectation[time] += paths[simu,time] / float(nb_simus) 

            plt.plot(times, integral_expectation, linewidth=.4, color='red')
            plt.plot(times, expectation, linewidth=.4, color='blue')

            plt.xlim(.0, self._T)
            plt.ylim(.0, max(max(expectation), max(integral_expectation)))

            plt.show()

        return paths, survival_prob

class PoissonProcess(LocalPoissonProcess):
    def __init__(self, intensity, T, N):
        intensity_funct = lambda t : intensity
        LocalPoissonProcess.__init__(self, intensity_funct, T, N)

#poisson_process = PoissonProcess(2.0, 10, 1000)
#paths, survival_prob = poisson_process.simulate(1000, True)

#poisson_process = PoissonProcess(2.0, 10, 1000)
#paths, survival_prob = poisson_process.simulate(1000, False, False, 10, True)

#intensity = lambda t : .5 * (t - 2.0) * (t - 2.0)
#poisson_process_local = LocalPoissonProcess(intensity, 10, 1000)
#paths, survival_prob = poisson_process_local.simulate(1000, False, False, 10, True)

blackKarasinski = BlackKarasinski(10.0, 1000, 0.2, 0.6, 1.5, 0.2)
intensity_paths = blackKarasinski.simulate(1000, True)
poisson_process_local = LocalPoissonProcess(intensity_paths, 10, 1000)
paths, survival_prob = poisson_process_local.simulate(1000, False, False, 10, True)