import itertools
import os.path
import random
import sys
from multiprocessing import Pool

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

peaks = [469.0,
         445.65254238930027,
         426.30193088574043,
         409.77768078680606,
         395.3584051526962,
         382.5679078760376,
         371.07528135454413,
         360.64122955849496,
         351.08708291938757,
         342.27591214268364,
         334.10045210230624,
         326.47511279834737,
         319.33050543059744,
         312.6095911274429,
         306.2649224317395,
         300.2565946984089,
         294.55073499662683,
         289.1183485032714,
         283.9344108300765,
         278.9771910720947,
         274.22768134291954,
         269.6691784338482,
         265.2869085748271,
         261.06775107595996,
         257.0]


def gaussian(x, amplitude, mean, stddev):
    return np.sqrt(1 / (2 * np.pi)) * amplitude * np.exp(-(x - mean) ** 2 / (2 * stddev ** 2))


def smooth(peaks, num, intensity):
    if intensity == 0:
        return np.zeros(25)
    Fer_sigma = 3.7
    FerSumo_sigma = 10.5
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


def model(_C, delta_time, K, indx):  # уравнения Смолуховского
    dC = np.zeros(_C.shape, dtype=np.float128)
    for [i, j, k, l] in indx:
        dC[i + k, j + l, :, :] += 1 / 2 * _C[i, j, :, :] * _C[k, l, :, :] * K[i, j, k, l, None, None]
        dC[i, j, :, :] -= K[i, j, k, l, None, None] * _C[i, j, :, :] * _C[k, l, :, :]
    return _C + dC * delta_time


def get_b():
    d = np.array([0.01402345, 0.02666863, 0.04298456, 0.05667676, 0.06925117,
                  0.07322666, 0.07076044, 0.06642211, 0.06316289, 0.06392605,
                  0.07256526, 0.06788632, 0.0620509, 0.0549545, 0.04503642,
                  0.0368877, 0.02996692, 0.02235332, 0.01716988, 0.01260734,
                  0.00951432, 0.00747281, 0.00571165, 0.00472962, 0.00399034])
    return d / np.sum(d)


def solver(C0, TIME, STEP, K, indx):
    prev = C0
    for t in tqdm(np.arange(0, TIME, STEP)):
        prev = model(prev, STEP, K, indx)
    return prev


def count(state, N, K, indx):
    C = np.zeros((N + 1, N + 1, 2, state.shape[1]), dtype=np.float128)

    state[0:2, :] /= np.sum(state[0:2, :], axis=0)
    state[2:, :] /= np.sum(state[2:, :], axis=0)

    C[0, 2, 0, :] = state[0, :]  # + state[7, :]
    C[1, 1, 0, :] = state[1, :]  # + state[8, :]

    C[2, 0, 1, :] = state[2, :]  # + state[7, :]
    C[1, 1, 1, :] = state[3, :]  # + state[8, :]

    c = solver(C, T, STEP, K, indx)

    cc = [np.zeros((N + 1, state.shape[1])), np.zeros((N + 1, state.shape[1]))]

    for j in range(N + 1):
        cc[0][j, :] = c[j, N - j, 0, :]
        cc[1][j, :] = c[j, N - j, 1, :]

    data = cc[0] + cc[1]

    for i in range(24):
        data[i, :] *= ((19.7) * i + (31.3) * (24 - i))

    for i in range(data.shape[1]):
        data[:, i] = smooth_profile(data[:, i] / np.sum(data[:, i], axis=0))

    data[data < 0.00001] = 0.00001
    delta = abs(data - get_b()[:, None])
    cc = delta ** 2 / data
    chi2 = np.sum(delta ** 2 / data, axis=0)
    delta = np.sum(delta ** 2 / (get_b())[:, None] / (get_b())[:, None], axis=0)
    return state, ((cc[0], cc[1]), data), chi2, delta


def get_K_matrix(N):
    K = np.zeros((N + 1, N + 1, N + 1, N + 1))
    for i in range(N + 1):
        for j in range(N + 1):
            for k in range(N + 1):
                for l in range(N + 1):
                    a = i + j
                    b = k + l
                    if a == 2 and b == 2:
                        K[i, j, k, l] = 1
                    if a == 4 and b == 2:
                        K[i, j, k, l] = 1
                    if a == 6 and b == 6:
                        K[i, j, k, l] = 1
                    if a == 12 and b == 12:
                        K[i, j, k, l] = 1
    indx = np.argwhere(K > 0.0)
    return K, indx

def is_correct(state):
    v1 = np.array([0, 1, 2, 3, 4, 5, 6])
    v2 = np.array([6, 5, 4, 3, 2, 1, 0])
    for i in range(state.shape[1]):
        s1 = np.dot(v1, state[:, i])
        s2 = np.dot(v2, state[:, i])
        delta = abs(s1 - s2) / max(0.00001, s2)
        if delta > 0.5:
            return False
    return True


if __name__ == '__main__':
    N = 24  # число субъединиц
    init_state = 4

    STEP = 0.1
    T = 300

    batch_size = 500
    best = 10
    multy = 30
    K, indx = get_K_matrix(N)
    batch = np.zeros((init_state, 1))

    # batch[:, 0] = c0
    # s0 = count(batch, N=N, K=K, indx=indx)
    # print(s0)
    res = {}
    # res = {s0[2][0]: s0[0].reshape(init_state)}
    min_data = None
    min_chi2 = None
    min_delta = None
    proc = 10
    directory = './plots14_first'
    if not os.path.exists(directory):
        os.makedirs(directory)

    for i in trange(1000):
        C = np.zeros((N + 1, N + 1))
        if len(res) == 0:
            batch = np.random.random((init_state, batch_size))
        else:
            batch = np.random.random((init_state, batch_size))
            for num, k in enumerate(res):
                for j in range(multy):
                    proc = 200
                    for l in range(init_state):
                        val = min(max((res[k][l] + random.random() / proc - 1 / 2 / proc), 0), 100)
                        batch[l, multy * num + j] = val
        state, data, chi2, delta = count(state=batch, N=N, K=K, indx=indx)
        dd = data[0]
        data = data[1]

        err_func = delta
        err_min = min_delta

        tmp = np.argsort(err_func)

        if min_data is None:
            min_data = data[:, tmp[0]]
            min_chi2 = chi2[tmp[0]]
            min_delta = delta[tmp[0]]
        elif err_func[tmp[0]] < err_min:
            min_data = data[:, tmp[0]]
            min_chi2 = chi2[tmp[0]]
            min_delta = delta[tmp[0]]
        for i in range(best):
            res[err_func[tmp[i]]] = state[:, tmp[i]]

        q = list(res.items())
        q.sort(key=lambda x: x[0])
        res = dict(q[0:best])
        tt = ",".join(itertools.repeat("{:4f}", init_state)).format(*q[0][1])
        text = """
        delta:      {:.4f}
        chi2:       {:.4f}
        state:      [{}]
        """.format(min_delta, min_chi2, tt)
        tqdm.write(text)
        fig, axs = plt.subplots(1, 1, figsize=(10, 10))
        axs.bar(range(0, N + 1), min_data,label=tt)
        axs.bar(range(0, N + 1), get_b(), label='experiment', alpha=0.5)

        # axs.scatter(range(0, N + 1), dd[0][:, tmp[0]],
        #                     label='c1')
        # axs.scatter(range(0, N + 1), dd[1][:, tmp[0]],
        #                     label='c2')
        axs.set(xlabel='i', ylabel='C')
        axs.legend(ncol=5)
        plt.savefig(
        os.path.join(directory, 'chi2: {:.3f}, delta:{:.3f}'.format(min_chi2,min_delta) + '_' + tt + '.png'))
        plt.clf()
        plt.close(fig)
