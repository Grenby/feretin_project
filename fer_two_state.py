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


def model(_C, delta_time, K, indx):  # уравнения Смолуховского
    dC = np.zeros(_C.shape)
    for [i, j, k, l] in indx:
        dC[i + k, j + l, :] += 1 / 2 * _C[i, j, :] * _C[k, l, :] * K[i, j, k, l]
        dC[i, j, :] -= K[i, j, k, l] * _C[i, j, :] * _C[k, l, :]
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
    d = np.array([0.02108793, 0.01815936, 0.02948557, 0.04665997, 0.06170539,
       0.0743812 , 0.08290072, 0.08712614, 0.08757545, 0.08468166,
       0.07801353, 0.0691656 , 0.05814158, 0.04805881, 0.03650942,
       0.02865245, 0.02117757, 0.01607109, 0.01193697, 0.0088738 ,
       0.00698342, 0.00577274, 0.00555825, 0.00557235, 0.00574904])

    # d /= np.sum(d)
    return d


def solver(C0, TIME, STEP, K, indx):
    prev = C0
    for t in np.arange(0, TIME, STEP):
        C_new = model(prev, STEP, K, indx)
        prev = C_new
    return prev


def count(state, N, K, indx):
    C = np.zeros((N + 1, N + 1, state.shape[1]))
    C[0, 1, :] = state[0, :]
    C[1, 0, :] = state[1, :]

    # C[:,:] /= np.sum(C, axis=(0,1))
    c = solver(C, T, STEP, K, indx)

    data = np.zeros((N + 1, state.shape[1]))
    for j in range(N + 1):
        data[j, :] = c[j, N - j, :]

    for i in range(24):
        data *= ((13 + 1 + 7 + 16) * i + (22 + 5 + 8 + 18) * (24 - i))

    for i in range(data.shape[1]):
        data[:, i] = smooth_profile(data[:, i])

    delta = abs(data - get_b()[:, None])
    chi2 = np.sum(delta ** 2 / get_b()[:, None], axis=0)
    delta = np.sum(delta ** 2 / (get_b() + 0.0001)[:, None] / (get_b() + 0.0001)[:, None], axis=0)
    return state, data, chi2, delta


def get_K_matrix(N):
    K = np.zeros((N + 1, N + 1, N + 1, N + 1))
    for i in range(N + 1):
        for j in range(N + 1):
            for k in range(N + 1):
                for l in range(N + 1):
                    a = i + j
                    b = k + l
                    # if a == 1 and b == 1:
                    #     K[i, j, k, l] = 1
                    # if a == 2 and b == 2:
                    #     K[i, j, k, l] = 1
                    # if (a == 4 and b == 2) or (a == 2 and b == 4):
                    #     K[i, j, k, l] = 1
                    if a == 6 and b == 6:
                        K[i, j, k, l] = 1
                    if a == 12 and b == 12:
                        K[i, j, k, l] = 1
    return K, np.argwhere(K > 0.01)


if __name__ == '__main__':
    N = 24  # число субъединиц

    thread = 1

    if len(sys.argv) != 2:
        thread = 1
    else:
        thread = int(sys.argv[1])

    STEP = 0.1
    T = 100

    min_delta = 1000000
    min_data = None

    batch_size = 100
    best = 10
    multy = 5

    # s = count([0.81773414, 0.6510148, 0.24049913, 0.18639066, 0.09990727, 0.50415137, 0.50301096])
    res = {}
    min_data = None
    min_chi2 = None
    min_delta = None

    K, indx = get_K_matrix(N)
    if 1 == 1:
        for i in trange(10000):
            C = np.zeros((N + 1, N + 1))
            if len(res) == 0:
                batch = np.random.random((2, batch_size))
            else:
                batch = np.random.random((2, batch_size))
                for num, k in enumerate(res):
                    for j in range(multy):
                        dd = 1
                        for l in range(2):
                            val = min(max((res[k][l] + random.random() / dd - 1 / 2 / dd), 0), 1)
                            batch[l, multy * num + j] = val
            KK = K.copy()
            for [i, j, k, l] in indx:
                KK[i, j, k, l] *= 10 * random.random()
            state, data, chi2, delta = count(state=batch, N=N, K=KK, indx=indx)

            err_func = chi2
            err_min = min_chi2

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
                res[err_func[tmp[i]]] = state[:, i]

            q = list(res.items())
            q.sort(key=lambda x: x[0])
            res = dict(q[0:best])

            text = """
                delta:      {:.4f}
                chi2:       {:.4f}
                state:      [{:.3f},{:.3f}]
            """.format(min_delta, min_chi2, *q[0][1].reshape(2))
            tqdm.write(text)

            fig, axs = plt.subplots(1, 1, figsize=(10, 10))
            axs.bar(range(0, N + 1), min_data)
            axs.bar(range(0, N + 1), get_b(), alpha=0.5)
            axs.set(xlabel='i', ylabel='C')
            # axs.legend(ncol=5)
            plt.savefig(
                'plots_2/' + str(thread) + 'chi2: {:.3f}, delta:{:.3f}'.format(min_chi2,
                                                                               min_delta) + '_{:.4f}_{:.4f}'.format(
                    *q[0][1])
                + '.png')
            plt.clf()
            plt.close(fig)

            # 0.677,0.084,0.426,0.926,0.860,0.396,0.108
