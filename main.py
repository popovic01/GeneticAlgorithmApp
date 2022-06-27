import json
import time
import threading

from pyquaternion import Quaternion
from population import *
import abb
import pygad
from collections import deque
import numpy as np
import matplotlib.pyplot as plt

speed = [] #brzine robota

def init_robots(config: json):
    rob_dict = {}

    for robot in config["robots"]:
        name = robot["name"]
        ip = robot["ip"]
        port = robot["port"]
        toolData = robot["tool_data"]
        wobj = robot["wobjSto"]
        speed = robot["speed"]

        #kreiranje robota klase Robot
        newRobot = abb.Robot(
            ip=ip,
            port_motion=port,
            port_logger=port + 1
        )

        #setovanje tool-a, wobj-a i brzine
        newRobot.set_tool(toolData)
        newRobot.set_workobject(wobj)
        newRobot.set_speed(speed)

        rob_dict[name] = newRobot

    return rob_dict

def single_fitness(solution):
    if len(solution) == 0: #ako nekom robotu nije dodeljena nijedna tacka
        return 0
    distance = DIST_MAP[solution[0], solution[-1]] #solution[-1] je poslednji element niza
    #racunanje distance izmedju prvog i poslednjeg targeta iz prosledjene liste, pomocu matrice DIST_MAP
    #npr. ako je lista [4,0,9,1] onda se racuna udaljenost od 4 do 1 (potrebno za zatvaranje trajektorije)

    #racunanje distance izmedju svaka dva targeta i dodavanje na distancu
    #distance od 4 do 0, od 0 do 9, od 9 do 1
    for idx1, idx2 in zip(solution[:-1], solution[1:]):
        distance += DIST_MAP[idx1, idx2]

    if solution[0] == 12: #u pitanju je 1. robot
        time = distance / speed[0] #t=s/v
    elif solution[0] == 13:  #u pitanju je 2. robot
        time = distance / speed[1] #t=s/v

    return time

def fitness(solution, solution_idx):
    totalTime = 0
    info = solution[0:Config.n_robots - 1]  #dobijamo listu elemenata - u nasem slucaju samo 1 element
    targets = solution[Config.n_robots - 1:] #lista targeta - brojevi od 0 do 11 (ako su svi targeti dohvatljivi)

    prevIndx = 0
    for i in info:
        subTargets = [12]
        for x in range(len(targets[prevIndx:i])):
            subTargets.append(targets[x])
        subTargets.append(12)
        totalTime += single_fitness(subTargets) #racunanje single fitnesa za targete za prvog robota
        prevIndx = i

    subTargets2 = [13]
    for x in range(prevIndx, len(targets)):
        subTargets2.append(targets[x])
    subTargets2.append(13)
    totalTime += single_fitness(subTargets2) #racunanje single fitnesa za targete za drugog robota
    #total distance je zbir pojedinacnih fitnesa svakog robota
    return -totalTime #sto je vreme krace, fitnes je veci (zato ide minus)

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
    #child_size je tuple (broj potrebne dece, duzina jedne jedinke)
    children = [] #niz dece
    idx = 0
    while len(children) != child_size[0]:
        p1_with_info = parents[idx % len(parents)].copy() #1. roditelj - cela jedinka
        p2_with_info = parents[(idx + 1) % len(parents)].copy() #2. roditelj - cela jedinka
        info1 = p1_with_info[0:Config.n_robots-1] #meta podatak 1. roditelja
        p1 = p1_with_info[Config.n_robots-1:] #targeti 1. roditelja
        p2 = p2_with_info[Config.n_robots-1:] #targeti 2. roditelja

        child_targets = ocx(p1, p2, len(p1))

        child = list()
        child.extend(info1)
        child.extend(child_targets)

        children.append(child)
        idx += 1

    return np.array(children)

def mutate(child, ga_inst):
    info = child[0:Config.n_robots - 1]  #lista meta podataka
    targets = child[Config.n_robots - 1:] #lista targeta

    id1, id2 = np.random.randint(0, len(targets)), np.random.randint(0, len(targets)) #2 random broja od 0 do broja targeta

    targets[id1], targets[id2] = targets[id2], targets[id1] #zamena mesta targetima sa generisanim indeksima

    probability = np.random.rand()
    new_info = list()
    #povecavanje ili smanjivanje meta podatka za 1
    if probability > 0.5:
        new_info = [np.clip(0, Config.n_targets-1, i+1) for i in info]
    else:
        new_info = [np.clip(0, Config.n_targets-1, i-1) for i in info]

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

    if len(solution) > 0:
        start_idx = solution[0]
        end_idx = solution[-1]
        t1 = TARGET_LIST2[start_idx]
        t2 = TARGET_LIST2[end_idx]
        x1, y1 = t1
        x2, y2 = t2
        plt.plot([x1, x2], [y1, y2])

    #plt.show()

def complete_plot(solution):
    info = solution[0:Config.n_robots - 1]
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

    quat = Quaternion(axis=[0, 1, 0], degrees=180)
    targetArray = [] #lista targeta
    homeTargets = [] #lista home pozicija

    for x in range(len(HOME_POSITION)):
        homeTargets.append([HOME_POSITION[x], quat.q])

    #dodajemo prvih n targeta (targeti za 1. robota), zatim m targeta (za 2. robota)
    for x in range(best_sol[0]):
        #dodajemo svaki target u niz targeta
        targetArray.append([targetsRobot1[x], quat.q])

    for x in range(Config.n_targets-best_sol[0]):
        #dodajemo svaki target u niz targeta
        targetArray.append([targetsRobot2[x], quat.q])

    t = threading.Thread(target=firstRobExecute, args=(robots, targetArray, homeTargets))
    t.start()
    secondRobExecute(robots, targetArray, homeTargets)

    #provera da li je doslo do kolizije
    collision = robots['ROB2'].collision(homeTargets[1]) #vraca collision ili no collision
    collisionBool = False
    if collision == 'collision':
        print('Došlo je do kolizije između robota')
        collisionBool = True
    elif collision == 'no collision':
        print('Nije došlo do kolizije između robota')
        collisionBool = False

    return collisionBool

def firstRobExecute(robots, targetArray, homeTargets):
    for x in range(Config.n_targets):
        #prvih n (meta podatak) targeta postvljamo 1. robotu
        if x < best_sol[0]-1:
            robots['ROB1'].set_cartesian(targetArray[x])
        elif x == best_sol[0]-1:
            robots['ROB1'].set_cartesian(targetArray[x])
            #da bi se 1. robot vratio u svoju pocetnu tacku
            robots['ROB1'].set_cartesian(targetArray[0])
            robots['ROB1'].set_cartesian(homeTargets[0])
            break

def secondRobExecute(robots, targetArray, homeTargets):
    for x in range(best_sol[0], Config.n_targets):
        if x > best_sol[0]-1:
            robots['ROB2'].set_cartesian(targetArray[x])

    #da bi se 2. robot vratio u svoju pocetnu tacku
    robots['ROB2'].set_cartesian(targetArray[best_sol[0]])
    robots['ROB2'].set_cartesian(homeTargets[1])

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

    for i in range(len(remove)):
            print('Target se izbacuje iz optimizacije:', VALID_TARGETS[i][0])
            VALID_TARGETS.remove(VALID_TARGETS[i])
            TARGET_LIST.remove(TARGET_LIST[i])
            TARGET_LIST2.remove(TARGET_LIST2[i])

    #update broja targeta
    Config.n_targets = len(VALID_TARGETS)
    print('Broj dohvatljivih targeta:', Config.n_targets)
    createPopulation()

    speed.append(robots['ROB1'].speed[0]) #brzina 1. robota
    speed.append(robots['ROB2'].speed[0]) #brzina 2. robota
    #print(speed)

def curr_solution(solution):
    targetsRobot1.clear()
    targetsRobot2.clear()
    for i in range(1, solution[0]+1):
        targetsRobot1.append(TARGET_LIST[solution[i]])

    for i in range(solution[0]+1, len(solution)):
        targetsRobot2.append(TARGET_LIST[solution[i]])

if __name__ == '__main__':
    check()

    ga_instance = pygad.GA(
        num_generations=Config.num_generations,
        initial_population=POPULATION,
        gene_type=int,

        mutation_type=mutate,
        mutation_probability=0.2,

        num_parents_mating=Config.parents_mating,
        keep_parents=20,

        fitness_func=fitness,
        crossover_type=crossover,
        save_solutions=True #sva resenja iz svake generacije su sacuvana u solutions
    )
    start = time.time()
    ga_instance.run()
    end = time.time()
    best_sol, best_sol_fitness, best_sol_idx = ga_instance.best_solution()

    #najbolje resenje
    #print(best_sol)
    #meta podatak
    #print(best_sol[0])
    targetsRobot1 = []
    targetsRobot2 = []

    curr_solution(best_sol)

    #print(targetsRobot1)
    #print(targetsRobot2)
    collisionBool = main()

    print(f"Vreme izvršavanja genetskog algoritma: {end - start:.5f}")
    complete_plot(best_sol)
    ga_instance.plot_fitness()

    if collisionBool == True:
        last_generation = [] #jedinke poslednje generacije
        x = Config.pop_size * Config.num_generations
        y = Config.pop_size * Config.num_generations + Config.pop_size
        for i in range(x, y):
            last_generation.append(ga_instance.solutions[i])
        fitnessList = list()  #fitnesi za poslednju generaciju

    while(collisionBool == True):
        fitnessList.clear()
        for i in range(len(last_generation)):
            fitnessList.append(fitness(last_generation[i], i))
            #racunanje fitnesa za svaku jedinku iz poslednje generacije
            #print(fitnessList[i])
        #print(np.max(fitnessList))
        max = np.max(fitnessList) #trenutni najbolji fitnes
        maxFitness = [] #lista indeksa iz liste koji imaju najbolji fitnes
        for i in range(len(fitnessList)):
            if fitnessList[i] == max:
                #print(fitnessList[i])
                maxFitness.append(i)
        #print("max fitness len")
        #print(len(maxFitness))
        j = 0
        for i in range(len(maxFitness)):
            if maxFitness[0] != maxFitness[i]:
                #ako nije prva iteracija, jer se pomeraju indeksi kad se element obrise
                j = j + 1
            last_generation.pop(maxFitness[i]-j) #brisanje trenutnih najboljih resenja
            fitnessList.pop(maxFitness[i]-j) #brisanje trenutnih najboljih fitnesa
            #print(maxFitness[i])
        #print("last gen, fitness list len")
        #print(len(last_generation))
        #print(len(fitnessList))
        max = np.max(fitnessList) #novi max fitness
        #print(max)
        indexMaxFitness = -1
        for i in range(len(fitnessList)):
            if fitnessList[i] == max:
                indexMaxFitness = i #indeks jedinke sa najvecim fitnesom
                break
        curr_solution(last_generation[indexMaxFitness])
        best_sol = last_generation[indexMaxFitness]
        if len(targetsRobot1) != 0:
            print("Novi targeti za 1. robota: ", targetsRobot1)
        collisionBool = main()
        complete_plot(best_sol)



