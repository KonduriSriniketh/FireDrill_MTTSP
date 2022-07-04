#!/usr/bin/env python3
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize
from pymoo.factory import get_algorithm, get_crossover, get_mutation, get_sampling
from pymoo.util.termination.default import SingleObjectiveDefaultTermination
import numpy as np
import math
import cmath
import random
from sympy import *
from pymoo.core.repair import Repair
from scipy.spatial.distance import cdist
from pymoo.core.problem import ElementwiseProblem

global F
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

class FireRescueSequence(ElementwiseProblem):

    def __init__(self,point_profile, **kwargs):
        self.n_waypoints, _ = point_profile.shape
        self.point_profile = point_profile
        super(FireRescueSequence, self).__init__(
        n_var = self.n_waypoints,
        n_obj = 1,
        xl = 0,
        xu = self.n_waypoints,
        type_var = int,
        **kwargs
        )

        # x message structure
        # first column start x pos
        # second column start y pos
        # third column velocity
        # fourth column yaw

    def _evaluate(self, x, out, *args, **kwargs):
        #print(x)
        robot_pt = np.zeros(2)
        robot_vel_abs = 2 # speed of the robot = 2 m/sec
        ini_pos = self.point_profile [:,:2]
        ini_vel = self.point_profile [:,2]
        ini_yaw = self.point_profile [:,3]
        out['F'] = self.estimate_route_length(self.n_waypoints, ini_pos, ini_vel, ini_yaw, robot_pt, robot_vel_abs, x)
        #print(out['F'])

    def estimate_route_length(self,n_waypoints, ini_pos, ini_vel, ini_yaw, robot_pt, robot_vel_abs, x):
        dt = 0
        ds = 0
        time = 0
        Distance = 0
        i = 0
        ini_pos_seq = np.zeros((ini_pos.shape), dtype=float)
        ini_vel_seq = np.zeros(ini_vel.shape)
        ini_yaw_seq = np.zeros(ini_yaw.shape)
        for j in range(0, n_waypoints):
            k = x[j]
            #print(k)
            a = ini_pos[k][0]
            ini_pos_seq[j,0] = ini_pos[k,0]
            ini_pos_seq[j,1] = ini_pos[k,1]
            ini_vel_seq[j] = ini_vel[k]
            ini_yaw_seq[j] = ini_yaw[k]
        """print("ini_pos")
        print(ini_pos)
        print("ini_pos_seq")
        print(ini_pos_seq)"""

        while (i < n_waypoints):
            target_pt = ini_pos_seq[i,:]
            target_vel = ini_vel_seq[i]
            target_yaw = ini_yaw_seq[i]
            """print("target_pt")
            print(target_pt)
            print("robot_pt")
            print(robot_pt)
            print("target_vel = ", target_vel)
            print("target_yaw = ", target_yaw)"""
            intersection_pt = self.estimate_intersection(target_pt, robot_pt, robot_vel_abs, target_vel, target_yaw)
            """print("intersection_pt")
            print(intersection_pt)
            print("-------------------")"""
            ds = self.euclidean_distance(robot_pt[0], robot_pt[1], intersection_pt[0], intersection_pt[1])
            dt = ds/robot_vel_abs
            if (i < n_waypoints -1):
                updated_pt = self.update_pt_fun(ini_pos_seq[i+1:,:], ini_vel_seq[i+1:], ini_yaw_seq[i+1:], dt)
            time = time + dt
            Distance = Distance + ds
            ini_pos_seq[i+1:,:] = updated_pt
            robot_pt = intersection_pt
            i = i+1
        print("distance = ", Distance)
        print("=====================")
        return (Distance)

    def update_pt_fun(self, pt, vel, yaw, time):
        updated_pt = pt
        len, _ = pt.shape
        for i in range(0, len):
            pt_vector = np.array([vel[i]*math.cos(yaw[i]), vel[i]*math.sin(yaw[i])])
            updated_pt[i,:] = pt_vector * time
        return updated_pt

    def estimate_intersection(self, target_ini_pt_vector, robot_ini_pt_vector, robot_vel, mag_target_vel, target_yaw):
        mag_target_ini_pt_vector = sqrt(target_ini_pt_vector[0]**2 + target_ini_pt_vector[1]**2)
        mag_robot_ini_pt_vector = sqrt(robot_ini_pt_vector[0]**2 + robot_ini_pt_vector[1]**2)
        target_vel_vector = np.array([mag_target_vel*math.cos(target_yaw), mag_target_vel*math.sin(target_yaw)])
        roots_time = self.findroots((mag_target_vel**2 - robot_vel**2), 2*(np.dot(target_ini_pt_vector, target_vel_vector)),mag_target_ini_pt_vector**2)
        #print ("roots_time = ", roots_time)
        if ((isinstance(roots_time[0], complex)) | (isinstance(roots_time[1], complex))):
            print("no intersection")
            return 0
        if ((roots_time[0] > 0) & (roots_time[1] > 0)):
            if (roots_time[0] < roots_time[1]):
                time = roots_time[0]
            else:
                time = roots_time[1]
            pt_intersection = target_vel_vector*time + target_ini_pt_vector
        elif ((roots_time[0] > 0) & (roots_time[1] <= 0)):
            time = roots_time[0]
            pt_intersection = target_vel_vector*time + target_ini_pt_vector
        elif ((roots_time[0] <= 0) & (roots_time[1] > 0)):
            time = roots_time[1]
            pt_intersection = target_vel_vector*time + target_ini_pt_vector
        elif ((roots_time[0] == 0) & (roots_time[1] == 0)):
            time = roots_time[0]
            pt_intersection = target_vel_vector*time + target_ini_pt_vector
        return pt_intersection

    def findroots(self, a, b, c):
        # calculating discriminant using formula

        a = round(a, 4)
        b = round(b, 4)
        c = round(c, 4)
        #print("a=",a, " b=", b, "c=", c)
        dis = (b**2) - (4*a*c)
        sqrt_val = round(math.sqrt(abs(dis)), 4)
        roots = np.array([0.0,0.0])
        # checking condition for discriminant
        if dis > 0:
            roots[0] = ((-b +(sqrt_val))/(2 * a))
            roots[1] = ((-b -(sqrt_val))/(2 * a))
        elif dis == 0:
            roots[0] = -b / (2 * a)
            roots[1] = -b / (2 * a)
        else:
            roots[0] = complex(-b/(2*a), sqrt_val)
            roots[1] = complex(-b/(2*a), -sqrt_val)
        return roots

    def euclidean_distance(self,x1,y1,x2,y2):
        dist = sqrt((x2 - x1)**2 + (y2 -y1)**2)
        return dist

def create_random_mttsp(n_points):
    profile = np.zeros((n_points,5))
    for i in range(0, n_points):
        profile[i][0] = round(random.uniform(-15, 15), 2)
        profile[i][1] = round(random.uniform(-14, 15), 2)
        profile[i][2] = round(random.uniform(0,0.5), 2)
        profile[i][3] = random.randrange(0,360)
    #return FireRescueSequence(point_profile=profile)
    return (profile)

def main():
    points = create_random_mttsp(10)
    problem = FireRescueSequence(point_profile=points)
    print(points)
    algorithm = GA(
    pop_size=20,
    sampling=get_sampling("perm_random"),
    crossover=get_crossover("perm_erx"),
    mutation=get_mutation("perm_inv"),
    repair=StartFromZeroRepair(),
    eliminate_duplicates=True
    )

    termination = SingleObjectiveDefaultTermination(n_last=20, n_max_gen=np.inf)
    res = minimize(
        problem,
        algorithm,
        termination,
        seed=1,
    )

    print(res.F)
    print("Traveling Time:", np.round(res.F[0], 3))
    from pymoo.problems.single.traveling_salesman import visualize
    visualize(problem, res.X)



if __name__ == '__main__':
    main()
