import os
import random
import sys
from multiprocessing import Pool

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

peaks = [290.0,
         263.2385273613206,
         241.05834530771193,
         222.1178133546881,
         205.59005873634518,
         190.92925289564693,
         177.75610079789726,
         165.79631501280318,
         154.8450997613735,
         144.74550306920816,
         135.3745748153793,
         126.63420948112461,
         118.44487179073201,
         110.74118228287088,
         103.46875542883345,
         96.58185147034607,
         90.04164435934115,
         83.81489946365548,
         77.87293316843673,
         72.19083693641039,
         66.7468234260823,
         61.52174697842029,
         56.498673507938605,
         51.66256373329372,
         47.0]


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
    dC = np.zeros(_C.shape)
    for [i, j, k, l] in indx:
        dC[i + k, j + l, :] += 1 / 2 * _C[i, j, :] * _C[k, l, :] * K[i, j, k, l, None]
        dC[i, j, :] -= K[i, j, k, l, None] * _C[i, j, :] * _C[k, l, :]
    return _C + dC * delta_time


def get_b():
    d = np.array([0.02108793, 0.01815936, 0.02948557, 0.04665997, 0.06170539,
                  0.0743812, 0.08290072, 0.08712614, 0.08757545, 0.08468166,
                  0.07801353, 0.0691656, 0.05814158, 0.04805881, 0.03650942,
                  0.02865245, 0.02117757, 0.01607109, 0.01193697, 0.0088738,
                  0.00698342, 0.00577274, 0.00555825, 0.00557235, 0.00574904])
    return d / d.sum()


def solver(C0, TIME, STEP, K, indx):
    prev = C0
    for _ in tqdm(np.arange(0, TIME, STEP)):
        prev = model(prev, STEP, K, indx)
    return prev


def count(state, N, K, indx):
    print(state.shape)
    C = np.zeros((N + 1, N + 1, state.shape[1]))
    C[0, 6, :] = state[0, :]
    C[1, 5, :] = state[1, :]
    C[2, 4, :] = state[2, :]
    C[3, 3, :] = state[3, :]
    C[4, 2, :] = state[4, :]
    C[5, 1, :] = state[5, :]
    C[6, 0, :] = state[6, :]
    for i in range(C.shape[2]):
        C[:, :, i] /= np.sum(C[:, :, i])
    c = solver(C, T, STEP, K, indx)

    data = np.zeros((N + 1, state.shape[1]))

    for j in range(N + 1):
        data[j, :] = c[j, N - j, :]

    for i in range(24):
        data[i, :] *=  ((19.7)*i + (31.3)*(24-i))

    for i in range(data.shape[1]):
        data[:, i] = smooth_profile(data[:, i] / np.sum(data[:, i], axis=0))
    data[data < 0.00001] = 0.00001
    delta = abs(data - get_b()[:, None])
    cc =delta ** 2 / data
    chi2 = np.sum(delta ** 2 / data, axis=0)
    delta = np.sum(delta ** 2 / (get_b())[:, None] / (get_b())[:, None], axis=0)
    return state, data, chi2, delta


def get_K_matrix(N):
    K = np.zeros((N + 1, N + 1, N + 1, N + 1))
    for i in range(N + 1):
        for j in range(N + 1):
            for k in range(N + 1):
                for l in range(N + 1):
                    a = i + j
                    b = k + l
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

    thread = 1

    if len(sys.argv) != 2:
        thread = 1
    else:
        thread = int(sys.argv[1])

    STEP = 0.1
    T = 1000

    batch_size = 500
    best = 20
    multy = 20

    res = {}

    min_data = None
    min_chi2 = None
    min_delta = None

    K, indx = get_K_matrix(N)
    state = np.array([[2.79720226e-01
                         , 1.67619945e-01
                         , 1.93646683e-01
                         , 1.66415834e-01
                         , 2.35770142e-02
                         , 7.71252763e-35
                         , 1.69020297e-01]])
    state /= np.sum(state)
    state = state.reshape(7,1)
    state, data, chi2, delta = count(state=state, N=N, K=K, indx=indx)
    # res = {delta[0]: state}
    directory = './plots'
    if not os.path.exists(directory):
        os.makedirs(directory)
    if 1 == 1:
        for i in trange(1000):
            if len(res) == 0:
                batch = np.random.random((7, batch_size))
            else:
                batch = np.random.random((7, batch_size))
                for num, k in enumerate(res):
                    for j in range(multy):
                        dd = 100
                        for l in range(7):
                            val = min(max((res[k][l] + random.random() / dd - 1 / 2 / dd), 0), 1)
                            batch[l, multy * num + j] = val
            state, data, chi2, delta = count(state=batch, N=N, K=K, indx=indx)
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
            print(min_data)

            for i in range(best):
                res[err_func[tmp[i]]] = state[:, tmp[i]]

            q = list(res.items())
            q.sort(key=lambda x: x[0])
            res = dict(q[0:best])

            if True:
                text = """
                    data:       {:.4f}
                    delta:      {:.4f}
                    chi2:       {:.4f}
                    state:      [{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f},{:.3f}]
                """.format(data[0, tmp[0]], min_delta, min_chi2, *q[0][1].reshape(7))
                tqdm.write(text)

                fig, axs = plt.subplots(1, 1, figsize=(10, 10))
                axs.bar(range(0, N + 1), min_data,
                        label='{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}'.format(*q[0][1]))
                axs.bar(range(0, N + 1), get_b(),
                        label='experiment', alpha=0.5)
                axs.set(xlabel='i', ylabel='C')
                axs.legend(ncol=5)
                plt.savefig(
                    './plots/' + str(thread) + 'chi2: {:.3f}, delta:{:.3f}'.format(min_chi2,
                                                                                           min_delta) + '_{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}_{:.4f}'.format(
                        *q[0][1])
                    + '.png')
                plt.clf()
                plt.close(fig)
