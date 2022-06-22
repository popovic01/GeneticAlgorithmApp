import numpy as np
import random
from scipy.spatial.distance import cdist

seed = 12345
np.random.seed(seed)
random.seed(seed)

class Config:
    n_targets = 12
    n_robots = 2
    pop_size = 300
    parents_mating = int(0.2 * pop_size)

# TARGET_LIST = [[-250.727, 32.339, 0], [-181.167, 31.793, 0], [-98.8, 42.013, 0], [-48.862, 82.15, 0],
#                 [-143.755, 108.275, 0], [-196.308, 153.932, 0], [-73.859, 154.937, 0], [-51.74, 231.333, 0],
#                 [-88.312, 277.863, 0], [-137.331, 280.576, 0], [-206.624, 267.382, 0], [-256.583, 213.549, 0]]
#
# TARGET_LIST2 = [[-250.727, 32.339], [-181.167, 31.793], [-98.8, 42.013], [-48.862, 82.15],
#                 [-143.755, 108.275], [-196.308, 153.932], [-73.859, 154.937], [-51.74, 231.333],
#                 [-88.312, 277.863], [-137.331, 280.576], [-206.624, 267.382], [-256.583, 213.549]]

TARGET_LIST = [
    [x, y, z] for x, y, z in zip(
        np.random.randint(-300, 0, size=Config.n_targets),
        np.random.randint(0, 300, size=Config.n_targets),
        [0]*Config.n_targets
    )
]

VALID_TARGETS = []

TARGET_LIST_COPY = []
for x in range(len(TARGET_LIST)):
    TARGET_LIST_COPY.append(TARGET_LIST[x].copy())
TARGET_LIST2 = []
for x in range(len(TARGET_LIST_COPY)):
    TARGET_LIST_COPY[x].pop()
    TARGET_LIST2.append(TARGET_LIST_COPY[x])

DIST_MAP = cdist(np.array(TARGET_LIST), np.array(TARGET_LIST))

#generisanje populacije
POPULATION = []
for i in range(Config.pop_size):
    #meta podaci - u nasem slucaju je to samo jedan (prvi) element
    metaInfo = [ np.random.randint(0, Config.n_targets) for _ in range(Config.n_robots-1) ]
    #lista targeta, ima ih 12, predstavljeni brojevima od 0 do 11, bez ponavljanja
    target_list = list(np.random.choice(list(range(Config.n_targets)), size=Config.n_targets, replace=False))

    newChromosome = list()
    newChromosome.extend(metaInfo)
    newChromosome.extend(target_list)
    POPULATION.append(newChromosome)





