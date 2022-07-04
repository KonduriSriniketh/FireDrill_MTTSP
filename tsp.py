#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from pymoo.core.repair import Repair

from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.factory import get_algorithm, get_crossover, get_mutation, get_sampling
from pymoo.problems.single.traveling_salesman import create_random_tsp_problem
from pymoo.util.termination.default import SingleObjectiveDefaultTermination

class StartFromZeroRepair(Repair):

    def _do(self, problem, pop, **kwargs):
        X = pop.get("X")
        I = np.where(X == 0)[1]

        for k in range(len(X)):
            i = I[k]
            x = X[k]
            _x = np.concatenate([x[i:], x[:i]])
            pop[k].set("X", _x)

        return pop

class TravelingSalesman(ElementwiseProblem):

    def __init__(self, cities, **kwargs):
        """
        A two-dimensional traveling salesman problem (TSP)
        Parameters
        ----------
        cities : numpy.array
            The cities with 2-dimensional coordinates provided by a matrix where where city is represented by a row.
        """
        n_cities, _ = cities.shape
        print = cities
        self.cities = cities
        self.D = cdist(cities, cities)

        super(TravelingSalesman, self).__init__(
            n_var=n_cities,
            n_obj=1,
            xl=0,
            xu=n_cities,
            type_var=int,
            **kwargs
        )

    def _evaluate(self, x, out, *args, **kwargs):
        out['F'] = self.get_route_length(x)

    def get_route_length(self, x):
        n_cities = len(x)
        dist = 0
        for k in range(n_cities - 1):
            i, j = x[k], x[k + 1]
            dist += self.D[i, j]

        last, first = x[-1], x[0]
        dist += self.D[last, first]  # back to the initial city
        return dist


def create_random_tsp_problem(n_cities, grid_width=100.0, grid_height=None, seed=None):
    if seed is not None:
        np.random.seed(seed)
    grid_height = grid_height if grid_height is not None else grid_width
    cities = np.random.random((n_cities, 2)) * [grid_width, grid_height]
    print(cities)
    print(cities.shape)
    return TravelingSalesman(cities)


def visualize(problem, x, fig=None, ax=None, show=True, label=True):
    with plt.style.context('ggplot'):

        if fig is None or ax is None:
            fig, ax = plt.subplots()

        # plot cities using scatter plot
        ax.scatter(problem.cities[:, 0], problem.cities[:, 1], s=250)
        if label:
            # annotate cities
            for i, c in enumerate(problem.cities):
                ax.annotate(str(i), xy=c, fontsize=10, ha="center", va="center", color="white")

        # plot the line on the path
        for i in range(len(x)):
            current = x[i]
            next_ = x[(i + 1) % len(x)]
            ax.plot(problem.cities[[current, next_], 0], problem.cities[[current, next_], 1], 'r--')

        fig.suptitle("Route length: %.4f" % problem.get_route_length(x))

        if show:
            fig.show()
def main():
    problem = create_random_tsp_problem(30, 100, seed=1)
    #print(problem)
    algorithm = GA(
        pop_size=20,
        sampling=get_sampling("perm_random"),
        crossover=get_crossover("perm_erx"),
        mutation=get_mutation("perm_inv"),
        repair=StartFromZeroRepair(),
        eliminate_duplicates=True
    )
    # if the algorithm did not improve the last 200 generations then it will terminate (and disable the max generations)
    termination = SingleObjectiveDefaultTermination(n_last=200, n_max_gen=np.inf)

    res = minimize(
        problem,
        algorithm,
        termination,
        seed=1,
    )
    print (res.F.shape)
    print("Traveling Time:", np.round(res.F[0], 3))
    from pymoo.problems.single.traveling_salesman import visualize
    visualize(problem, res.X)

if __name__ == '__main__':
    main()
