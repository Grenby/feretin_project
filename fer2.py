import random
import sys
from multiprocessing import Pool

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

plt.rc('legend', fontsize=8)  # legend fontsize
plt.rcParams.update({'font.size': 14})

N = 24  # число субъединиц
K = np.zeros((N + 1, N + 1, N + 1, N + 1))
for i in range(N + 1):
    for j in range(N + 1):
        for k in range(N + 1):
            for l in range(N + 1):
                a = i + j
                b = k + l
                # if a == 1 and b == 1:
                #   K[i,j,k,l]=1
                # if a == 2 and b == 2:
                #   K[i,j,k,l]=1
                # if (a == 4 and b == 2) or (a == 2 and b == 4):
                #   K[i,j,k,l]=1
                if a == 6 and b == 6:
                    K[i, j, k, l] = 1
                if a == 12 and b == 12:
                    K[i, j, k, l] = 1

indx = np.argwhere(K > 0.01)

peaks = [0.8066787214453491, 0.7286908212636174, 0.6640536524738351, 0.6088574255963066, 0.560692479862323,
         0.517968169510449, 0.4795791583843122, 0.4447261219839642, 0.4128122473841507, 0.38338014738955173,
         0.3560715225191292, 0.33060047507555473, 0.3067352244977619, 0.28428524289616, 0.2630920407421098,
         0.24302232388731304, 0.22396294601870886, 0.20581705524736288, 0.18850106219379073, 0.1719423787555041,
         0.15607751257404356, 0.14085066964413634, 0.1262125008964556, 0.11211917907276119, 0.09853160263657085]


def gaussian(x, amplitude, mean, stddev):
    return np.sqrt(1 / (2 * np.pi)) * amplitude * np.exp(-(x - mean) ** 2 / (2 * stddev ** 2))


def smooth(peaks, num, intensity):
    if intensity == 0:
        return np.zeros(25)
    Fer_sigma = 0.011
    FerSumo_sigma = 0.024
    sigma = FerSumo_sigma * (24 - num) / 24.0 + Fer_sigma * num / 24.0
    peak_center = peaks[num]
    smoothed_peak = np.zeros(25)
    for i in range(1, 24):
        smoothed_peak[i] = gaussian(peaks[i], intensity, peak_center, sigma) * (peaks[i - 1] - peaks[i + 1]) / 2
    smoothed_peak[0] = gaussian(peaks[0], intensity, peak_center, sigma) * (peaks[0] - peaks[1])
    smoothed_peak[24] = gaussian(peaks[24], intensity, peak_center, sigma) * (peaks[23] - peaks[24])
    return smoothed_peak / sum(smoothed_peak) * intensity


def smooth_profile(p_24):
    result = np.zeros(25)
    for i in range(0, 25):
        result += smooth(peaks, i, p_24[i])
    return result


def model(_C, delta_time):  # уравнения Смолуховского
    dC = np.zeros((N + 1, N + 1))
    for [i, j, k, l] in indx:
        dC[i + k, j + l] += 1 / 2 * _C[i, j] * _C[k, l] * K[i, j, k, l]
        dC[i, j] -= K[i, j, k, l] * _C[i, j] * _C[k, l]
    return _C + dC * delta_time


def get_b():
    b = np.zeros(25)
    b[0] = 0.015
    b[1] = 0.025
    b[2] = 0.038
    b[3] = 0.05
    b[4] = 0.061
    b[5] = 0.068
    b[6] = 0.068
    b[7] = 0.065
    b[8] = 0.061
    b[9] = 0.059
    b[10] = 0.059
    b[11] = 0.061
    b[12] = 0.055
    b[13] = 0.052
    b[14] = 0.047
    b[15] = 0.043
    b[16] = 0.036
    b[17] = 0.03
    b[18] = 0.025
    b[19] = 0.02
    b[20] = 0.017
    b[21] = 0.015
    b[22] = 0.011
    b[23] = 0.009
    b[24] = 0.007
    b = [0.0040782, 0.01298694, 0.02030641, 0.02340803, 0.02375074, 0.03095877,
         0.04973027, 0.06252962, 0.06053318, 0.0524339, 0.05017999, 0.05999055,
         0.06524802, 0.05391921, 0.03848055, 0.02961201, 0.02858626, 0.02762937,
         0.01945947, 0.01024822, 0.00584425, 0.00461172, 0.00398311, 0.00234085,
         0.00058389]
    b = np.array(b)
    b /= np.sum(b)
    return b


def solver(C0, TIME, STEP):
    prev = C0
    # times = [-STEP]
    for t in np.arange(0, TIME, STEP):
        C_new = model(prev, STEP)
        prev = C_new
    return prev


def count(state):
    C = np.zeros((N + 1, N + 1))
    C[0, 6] = state[0]
    C[1, 5] = state[1]
    C[2, 4] = state[2]
    C[3, 3] = state[3]
    C[4, 2] = state[4]
    C[5, 1] = state[5]
    C[6, 0] = state[6]
    c = solver(C, T, STEP)

    data = np.zeros(N + 1)
    for j in range(N + 1):
        data[j] = c[j, N - j]
    # data = smooth_profile(data)
    data /= sum(data)
    return np.sum(((data - get_b()) ** 2)/get_b()/get_b()), state


if __name__ == '__main__':
    if len(sys.argv) != 2:
        thread = 4
    else:
        thread = int(sys.argv[1])

    STEP = 0.1
    T = 100
    b = get_b()
    min_delta = 1000000
    min_data = None
    batch_size = 32
    best = 5
    multy = 6
    # C[0,6] = 0.81773414
    # C[1,5] = 0.6510148
    # C[2,4] = 0.24049913
    # C[3,3] = 0.18639066
    # C[4,2] = 0.09990727
    # C[5,1] = 0.50415137
    # C[6,0] = 0.50301096
    s = count([0.81773414,0.6510148,0.24049913,0.18639066,0.09990727,0.50415137,0.50301096])
    res = {s[0]:s[1]}
    print(res)
    with Pool(thread) as p:
        for i in trange(100):
            C = np.zeros((N + 1, N + 1))
            if len(res) == 0:
                batch = np.random.random((batch_size, 7))
            else:
                batch = np.random.random((batch_size, 7))
                for num, k in enumerate(res):
                    for j in range(multy):
                        for l in range(7):
                            batch[multy * num + j, l] = min(max(res[k][l] + random.random() / 100 - 1 / 200, 0), 1)
            tmp = p.map(count, batch)
            for t in tmp:
                print(t[0],np.sum(abs(t[1] - np.array([0.81773414,0.6510148,0.24049913,0.18639066,0.09990727,0.50415137,0.50301096]))))
                res[t[0]] = t[1]
            q = list(res.items())
            q.sort(key=lambda x: x[0])
            res = dict(q[0:best])

            q = list(res.items())
            q.sort(key=lambda x: x[0])
            q = q[0:1]
            print('delta:', q[0][0])
            print('c:', q[0][1])
