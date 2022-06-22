import json
import time
from pyquaternion import Quaternion
from population import *
import abb
import pygad
from collections import deque
import numpy as np
import matplotlib.pyplot as plt

def init_robots(config: json):
    rob_dict = {}

    for robot in config["robots"]:
        name = robot["name"]
        ip = robot["ip"]
        port = robot["port"]
        toolData = robot["tool_data"]
        wobj = robot["wobjSto"]

        #kreiranje robota klase Robot
        newRobot = abb.Robot(
            ip=ip,
            port_motion=port,
            port_logger=port + 1
        )

        #setovanje tool-a i wobj-a
        newRobot.set_tool(toolData)
        newRobot.set_workobject(wobj)

        rob_dict[name] = newRobot
        #print(rob_dict)

    return rob_dict

def single_fitness(solution):
    if len(solution) <= Config.n_targets//2 - 1:
        return 0
    distance = DIST_MAP[solution[0], solution[-1]]

    for idx1, idx2 in zip(solution[:-1], solution[1:]):
        distance += DIST_MAP[idx1, idx2]

    return distance

def fitness(solution, solution_idx):
    totalDistance = 0
    info = solution[0:Config.n_robots - 1]  # dobijamo listu elemenata
    targets = solution[Config.n_robots - 1:]

    prevIndx = 0
    for i in info:
        subTargets = targets[prevIndx:i]
        totalDistance += single_fitness(subTargets)
        prevIndx = i

    totalDistance += single_fitness(targets[prevIndx:])

    return -totalDistance

def ocx(p1, p2, size):
    """ Ordered cycle crossover"""

    indx1 = np.random.randint(0, size)
    indx2 = np.random.randint(0, size)
    if indx2 < indx1:
        indx1, indx2 = indx2, indx1

    child = deque(p1[indx1:indx2])

    while len(child) != size:
        if p2[indx2] not in child:
            child.append(p2[indx2])
        indx2 = (indx2 + 1) % size

    child.rotate(indx1)
    return child

def crossover(parents, child_size, ga_inst):
    # child_size je tuple (broj potrebne dece, duzina jedne jedinke)
    children = []
    idx = 0
    while len(children) != child_size[0]:
        p1_with_info = parents[idx % len(parents)].copy()
        p2_with_info = parents[(idx + 1) % len(parents)].copy()
        info1 = p1_with_info[0:Config.n_robots-1]
        #info2 = p2_with_info[0:Config.n_robots-1]
        p1 = p1_with_info[Config.n_robots-1:]
        p2 = p2_with_info[Config.n_robots-1:]

        child_targets = ocx(p1, p2, len(p1))

        child = list()
        child.extend(info1)
        child.extend(child_targets)

        children.append(child)
        idx += 1

    return np.array(children)

def mutate(child, ga_inst):
    info = child[0:Config.n_robots - 1]  # dobijamo listu elemenata
    targets = child[Config.n_robots - 1:]

    id1, id2 = np.random.randint(0, len(targets)), np.random.randint(0, len(targets))

    targets[id1], targets[id2] = targets[id2], targets[id1]

    probability = np.random.rand()
    new_info = list()
    if probability > 0.5:
        new_info = [ np.clip(0, Config.n_targets-1, i+1) for i in info]
    else:
        new_info = [ np.clip(0, Config.n_targets-1, i-1) for i in info]

    mutated = list()
    mutated.extend(new_info)
    mutated.extend(targets)
    return mutated

def plot(solution):
    xs = [x for x, y in TARGET_LIST2]
    ys = [y for x, y in TARGET_LIST2]
    plt.scatter(xs, ys)

    for idx1, idx2 in zip(solution[:-1], solution[1:]):
        t1 = TARGET_LIST2[idx1]
        t2 = TARGET_LIST2[idx2]
        x1, y1 = t1
        x2, y2 = t2
        plt.plot([x1, x2], [y1, y2])

    start_idx = solution[0]
    end_idx = solution[-1]
    t1 = TARGET_LIST2[start_idx]
    t2 = TARGET_LIST2[end_idx]
    x1, y1 = t1
    x2, y2 = t2
    plt.plot([x1, x2], [y1, y2])

    #plt.show()

def complete_plot(solution):
    info = solution[0:Config.n_robots - 1]  # dobijamo listu elemenata
    targets = solution[Config.n_robots - 1:]

    prevIndx = 0
    for i in info:
        subTargets = targets[prevIndx:i]
        plot(subTargets)
        prevIndx = i

    plot(targets[prevIndx:])

    plt.show()

def main():
    file = open("config.json")
    config_json = json.load(file)
    robots = init_robots(config_json)

    #koordinate uzmemo u odnosu na parent k.s. i zamenimo x i y koordinate, a ispred prve stavimo suprotan predznak
    # xyzArray = [[-250.727, 32.339, 0], [-181.167, 31.793, 0], [-98.8, 42.013, 0], [-48.862, 82.15, 0],
    #             [-143.755, 108.275, 0], [-196.308, 153.932, 0], [-73.859, 154.937, 0], [-51.74, 231.333, 0],
    #             [-88.312, 277.863, 0], [-137.331, 280.576, 0], [-206.624, 267.382, 0], [-256.583, 213.549, 0]]
    quat = Quaternion(axis=[0, 1, 0], degrees=180)
    targetArray = []
    #dodajemo prvih 6 targeta (targeti za 1. robota), zatim 2. 6 targeta (za 2. robota)
    for x in range(best_sol[0]):
        #dodajemo svaki target u niz targeta
        #targetArray.append([xyzArray[x], quat.q])
        targetArray.append([targetsRobot1[x], quat.q])

    for x in range(Config.n_targets-best_sol[0]):
        #dodajemo svaki target u niz targeta
        targetArray.append([targetsRobot2[x], quat.q])

    for x in range(Config.n_targets):
        #prvih n (meta podatak) targeta postvljamo 1. robotu
        if x < best_sol[0]-1:
            robots['ROB1'].set_cartesian(targetArray[x])
        elif x == best_sol[0]-1:
            robots['ROB1'].set_cartesian(targetArray[x])
            #da bi se 1. robot vratio u svoju pocetnu tacku
            robots['ROB1'].set_cartesian(targetArray[0])
        else:
            robots['ROB2'].set_cartesian(targetArray[x])

    #da bi se 2. robot vratio u svoju pocetnu tacku
    robots['ROB2'].set_cartesian(targetArray[best_sol[0]])

def check():
    file = open("config.json")
    config_json = json.load(file)
    robots = init_robots(config_json)

    quat = Quaternion(axis=[0, 1, 0], degrees=180)
    #za svih 12 targeta, provericemo da li roboti mogu da ih dohvate
    #ukoliko ni jedan ne moze, izbacujemo ih iz optimizacije
    for x in range(Config.n_targets):
        #print(TARGET_LIST[x])
        VALID_TARGETS.append([TARGET_LIST[x], quat.q])

    #targets for removing - u listi ce biti indeksi targeta koji neki od robota ne moze da dohvati
    remove = []

    for x in range(Config.n_targets):
        isReachable = robots['ROB2'].is_reachable(VALID_TARGETS[x])
        if isReachable == 'false':
            print('Target nije dohvatljiv za drugog robota:', VALID_TARGETS[x][0])
            remove.append(x)

    for y in range(Config.n_targets):
        isReachable = robots['ROB1'].is_reachable(VALID_TARGETS[y])
        if isReachable == 'false':
            print('Target nije dohvatljiv za prvog robota:', VALID_TARGETS[y][0])
            remove.append(y)

    for i in range(len(remove)-1):
        if remove[i+1] == remove[i]:
            print('Target nije dohvatljiv za oba robota:', VALID_TARGETS[i][0])
            VALID_TARGETS.remove(VALID_TARGETS[i])
            TARGET_LIST.remove(TARGET_LIST[i])

    #update broja targeta
    Config.n_targets = len(VALID_TARGETS)
    print('Broj dohvatljivih targeta:', Config.n_targets)

if __name__ == '__main__':
    check()

    ga_instance = pygad.GA(
        num_generations=1000,
        initial_population=POPULATION,
        gene_type=int,

        mutation_type=mutate,
        mutation_probability=0.2,

        #parent_selection_type="sss",
        num_parents_mating=Config.parents_mating,
        keep_parents=20,

        fitness_func=fitness,
        crossover_type=crossover
    )
    start = time.time()
    ga_instance.run()
    end = time.time()
    best_sol, best_sol_fitness, best_sol_idx = ga_instance.best_solution()

    #najbolje resenje
    #print(best_sol)
    #meta podatak
    #print(best_sol[0])
    #print(TARGET_LIST)
    #targeti koje izvrsava 1. robot
    #print(best_sol[1:best_sol[0]+1])
    #targeti koje izvrsava 2. robot
    #print(best_sol[best_sol[0]+1:])
    targetsRobot1 = []
    targetsRobot2 = []

    for i in range(1, best_sol[0]+1):
        targetsRobot1.append(TARGET_LIST[best_sol[i]])

    for i in range(best_sol[0]+1, len(best_sol)):
        targetsRobot2.append(TARGET_LIST[best_sol[i]])

    print(targetsRobot1)
    print(targetsRobot2)
    main()

    print(f"Time: {end - start:.5f}")
    complete_plot(best_sol)
    ga_instance.plot_fitness()


