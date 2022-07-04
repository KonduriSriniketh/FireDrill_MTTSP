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
import matplotlib.pyplot as plt

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
        self.robot_intersection_pt_list = np.empty((0,self.n_waypoints,2),float)
        self.final_distance = 1000
        self.final_output_sequence = np.arange(self.n_waypoints)
        print("self.final_output_sequence = ", self.final_output_sequence)
        self.final_robot_intersection_pt = np.empty((self.n_waypoints,2), float)

        super(FireRescueSequence, self).__init__(
        n_var = self.n_waypoints,
        n_obj = 1,
        xl = 0,
        xu = self.n_waypoints,
        type_var = int,
        **kwargs
        )

        # point profile message structure
        # first column start x pos
        # second column start y pos
        # third column velocity
        # fourth column yaw in degrees

    def _evaluate(self, x, out, *args, **kwargs):
        print("x = ",x)
        robot_pt = np.zeros(2)
        robot_vel_abs = 0.4 # speed of the robot = 2 m/sec
        ini_pos = self.point_profile [:,:2]
        ini_vel = self.point_profile [:,2]
        ini_yaw = self.point_profile [:,3]
        out['F'] = self.estimate_route_length(self.n_waypoints, ini_pos, ini_vel, ini_yaw, robot_pt, robot_vel_abs, x)
        print("self.final_output_sequence = ",self.final_output_sequence)
        print("self.final_distance = ", self.final_distance)
        print("-------------------------------------")
        #print(out['F'])

    def estimate_route_length(self,n_waypoints, ini_pos, ini_vel, ini_yaw, robot_pt, robot_vel_abs, x):
        #initializations
        dt = 0
        ds = 0
        time = 0
        Distance = 0
        i = 0
        ini_pos_seq = np.zeros((ini_pos.shape), dtype=float)
        ini_vel_seq = np.zeros(ini_vel.shape)
        ini_yaw_seq = np.zeros(ini_yaw.shape)
        #arrange the way points in the order of the generated x sequence
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
        robot_intersection_pts = np.empty((0,2), float)
        # find all the intersection points for the sequence x
        i = 0
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
            time = time + dt
            Distance = Distance + ds

            robot_intersection_pts = np.vstack((robot_intersection_pts, intersection_pt))
            robot_pt = intersection_pt
            if (i < n_waypoints -1):
                updated_pt = self.update_pt_fun(ini_pos_seq[i+1,:], ini_vel_seq[i+1], ini_yaw_seq[i+1], time, robot_pt)
                ini_pos_seq[i+1,:] = updated_pt
                #print(updated_pt)
            #print(intersection_pt)
            i = i+1
        self.robot_intersection_pt_list = np.vstack((self.robot_intersection_pt_list, [robot_intersection_pts]))
        """print("robot_intersection_pts = ")
        print(robot_intersection_pts)
        print("=====================")"""
        """print("distance = ", Distance)
        print("=====================")"""
        if (Distance < self.final_distance):
            self.final_distance = Distance
            self.final_output_sequence = x
            self.final_robot_intersection_pt = robot_intersection_pts
        #print("self.final_robot_intersection_pt ")
        #print(self.final_robot_intersection_pt)
        #print("=====================")
        print("Distance =  " ,Distance)
        return (Distance)

    def estimate_intersection(self, target_ini_pt_vector, robot_ini_pt_vector, robot_vel, mag_target_vel, target_yaw):
        #magnitude of the position vector from origin
        mag_target_ini_pt_vector = sqrt(target_ini_pt_vector[0]**2 + target_ini_pt_vector[1]**2)
        mag_robot_ini_pt_vector = sqrt(robot_ini_pt_vector[0]**2 + robot_ini_pt_vector[1]**2)

        #tanslation of the vector in robot frame with robot postion as A_origin
        translated_target_vector_ini_pt = self.translate_pt_from_origin_to_A(robot_ini_pt_vector, target_ini_pt_vector)
        mag_translated_target_ini_pt_vector = sqrt(translated_target_vector_ini_pt[0]**2 + translated_target_vector_ini_pt[1]**2)

        target_vel_vector = np.array([mag_target_vel*math.cos(math.radians(target_yaw)), mag_target_vel*math.sin(math.radians(target_yaw))])
        roots_time = self.findroots((mag_target_vel**2 - robot_vel**2), 2*(np.dot(translated_target_vector_ini_pt, target_vel_vector)),mag_translated_target_ini_pt_vector**2)
        #print ("roots_time = ", roots_time)
        pt_intersection = np.empty(2, dtype = float)

        if ((isinstance(roots_time[0], complex)) | (isinstance(roots_time[1], complex))):
            print("no intersection")
            return 0
        if ((roots_time[0] > 0) & (roots_time[1] > 0)):
            if (roots_time[0] < roots_time[1]):
                time = roots_time[0]
            else:
                time = roots_time[1]
            pt_intersection = target_vel_vector*time + target_ini_pt_vector
            #print("if-1, pt_inter = ",pt_intersection )
        elif ((roots_time[0] > 0) & (roots_time[1] <= 0)):
            time = roots_time[0]
            pt_intersection = target_vel_vector*time + target_ini_pt_vector
            #print("if-2, pt_inter = ",pt_intersection )
        elif ((roots_time[0] <= 0) & (roots_time[1] > 0)):
            time = roots_time[1]
            pt_intersection = target_vel_vector*time + target_ini_pt_vector
            #print("if-3, pt_inter = ",pt_intersection )
        elif ((roots_time[0] == 0) & (roots_time[1] == 0)):
            time = roots_time[0]
            pt_intersection = target_vel_vector*time + target_ini_pt_vector
            #print("if-4, pt_inter = ",pt_intersection )
        return (pt_intersection)

    def update_pt_fun(self, pt, vel, yaw, time, robot_origin):
        updated_pt = pt
        # translation of initial postion vector into robot_origin cordinate frame
        transalted_postion_vector = self.translate_pt_from_origin_to_A(robot_origin, pt)
        #calulating velocity vector which has moved for t = time amount
        vel_vector = np.array([vel*math.cos(math.radians(yaw)), vel*math.sin(math.radians(yaw))])
        # calculating the resultant postion vector after t = time amount in robot_origin frame
        updated_pt = transalted_postion_vector + vel_vector * time
        #convert updated_pt back to (0,0) origin cordinate system
        updated_pt = updated_pt + robot_origin
        return updated_pt

    def translate_pt_from_origin_to_A(self, A_origin, pt):
        translated_pt = np.array([0.0,0.0])
        translated_pt[0] = pt[0] - A_origin[0]
        translated_pt[1] = pt[1] - A_origin[1]
        return (translated_pt)

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
    profile = np.zeros((n_points,4))
    for i in range(0, n_points):
        profile[i][0] = round(random.uniform(-10, 10), 2)
        profile[i][1] = round(random.uniform(-10, 10), 2)
        profile[i][2] = round(random.uniform(0,0.2), 2)
        profile[i][3] = random.randrange(0,360)
    #return FireRescueSequence(point_profile=profile)
    return (profile)

def main():
    points = create_random_mttsp(10)
    problem = FireRescueSequence(point_profile=points)

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
    print(points)
    print(res.F)
    print(problem.point_profile)
    fig=None
    ax=None
    show=True
    label=True
    if fig is None or ax is None:
        fig, ax = plt.subplots(1)

    plt.scatter(problem.point_profile[:, 0],
                problem.point_profile[:, 1],
                color = 'blue')
    plt.scatter(problem.final_robot_intersection_pt[:,0],
                problem.final_robot_intersection_pt[:,1],
                color = 'red' )
    plt.plot(problem.final_robot_intersection_pt[:,0],
             problem.final_robot_intersection_pt[:,1],
             color = 'gray')
    n = np.arange(0,problem.point_profile.shape[0])
    for i, txt in enumerate(n):
        k = problem.final_output_sequence[i]
        plt.plot([problem.point_profile[k,0],problem.final_robot_intersection_pt[i,0]],
                 [problem.point_profile[k,1],problem.final_robot_intersection_pt[i,1]],
                 color = 'green',
                 linestyle='dashed')
        plt.annotate(txt,
                     (problem.point_profile[i,0], problem.point_profile[i,1])
                    )
        plt.arrow(problem.point_profile[i,0],problem.point_profile[i,1],
                  dx = math.cos(math.radians(problem.point_profile[i,3])),
                  dy = math.sin(math.radians(problem.point_profile[i,3])),
                  width=.08,
                  facecolor='blue',
                  edgecolor='none')
        plt.annotate(k,
                     (problem.final_robot_intersection_pt[i,0], problem.final_robot_intersection_pt[i,1])
                    )
    if show:
        print("TRUE")
        plt.show()

if __name__ == '__main__':
    main()
