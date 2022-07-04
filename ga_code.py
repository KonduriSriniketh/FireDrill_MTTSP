#!/usr/bin/env python3
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.factory import get_algorithm, get_crossover, get_mutation, get_sampling
from pymoo.util.termination.default import SingleObjectiveDefaultTermination
import numpy as np
import math
from sympy import *
from pymoo.core.repair import Repair
from scipy.spatial.distance import cdist
from pymoo.core.problem import ElementwiseProblem

class FireRescueSequence(ElementwiseProblem):

    def __init__(self,Human_Profile, **kwargs):
        n_human, _ = Human_Profile.shape
        super(FireRescueSequence).__init__(
        n_var = n_human,
        n_obj = 1,
        xl = 0,
        xu = n_human,
        type_var = int,
        **kwargs
        )

    def _evaluate(self, x, out, *args, **kwargs):
        vel_n_yaw = x[2:,2:]
        co_orditanes = x[2:,:2]
        robot_init_pos = x[1,:2]
        staircase_init_pos = x[2,:2]
        dist_from_robot = self.get_distance_from_robot(x)
        dist_from_staircase = self.get_distance_from_staircase(x)
        velocity_direction = x[2:,2]
        f1 = dist_from_robot + dist_from_staircase
        out['F'] = f1

    def estimate_position(self, points, vel_n_yaw):
        t = 10
        len,_ = points.shape
        est_points = np.zeros((points.shape))
        line_eq = self.get_line_eq(points,vel_n_yaw)
        for i in range(len):
            if (vel_n_yaw[i] > 90 & vel_n_yaw[i] < 270):
                est_points[i][0] = points[i][0] + t*(vel_n_yaw[i][0])*(math.cos(math.radians(vel_n_yaw[i][1])))
                est_points[i][1] = line_eq[i][0]*est_points[i][0] + line_eq[i][1]
            elif (vel_n_yaw[i] <= 90 | (vel_n_yaw[i] >= 270 & vel_n_yaw[i] <=359)):
                est_points[i][0] = points[i][0] - t*(vel_n_yaw[i][0])*(math.cos(math.radians(vel_n_yaw[i][1])))
                est_points[i][1] = line_eq[i][0]*est_points[i][0] + line_eq[i][1]
        return (est_points)

    def predict(self, staircase_pos, robot_pos, ini_pos, est_pos):
        for i range()
        return 0

    def euclidean_distance(x1,y1,x2,y2):
        dist = (sqrt((x2 - x1)**2 + (y2 -y1)**2))
        return dist

    def get_distance_from_robot(self, x):
        n_humans = len(x) - 2
        dist = np.array([])
        for i in range(0,n_humans):
            d = euclidean_distance(x[i+2][0],x[i+2][1],x[0][0],x[0][1])
            dist.append(d)
        return dist

    def get_distance_from_staircase(self, x):
        n_humans = len(x) - 2
        dist = np.array([])
        for i in range(0,n_humans):
            d = euclidean_distance(x[i+2][0],x[i+2][1],x[1][0],x[1][1])
            dist.append(d)
        return dist

    def get_line_eq(self, points, vel_n_yaw): #get m and c in y=mx+c
        len,_ = points.shape
        l_eq = np.zeros((points.shape))
        for i in range(len):
            m = math.tan(math.radians(vel_n_yaw[i][1]))
            c = points[i][1] - m*points[i][0]
            l_eq[i][0], l_eq[i][1] = m, c
        return l_eq

def create_random_rescue_problem(n_humans, n_staircase):
    profile = np.zeros((n_humans+2, 4))
    profile[0,:2] = [2,2]
    profile[1,:2] = [0,3]
    for i in range(2,n_humans+2):
        profile[i][0] = random()
        profile[i][1] =
        profile[i][2] =
        profile[i][3] =
