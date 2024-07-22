import itertools
import os.path
import random
import sys
from multiprocessing import Pool

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

peaks = [0.8066787214453491, 0.7286908212636174, 0.6640536524738351, 0.6088574255963066, 0.560692479862323,
         0.517968169510449, 0.4795791583843122, 0.4447261219839642, 0.4128122473841507, 0.38338014738955173,
         0.3560715225191292, 0.33060047507555473, 0.3067352244977619, 0.28428524289616, 0.2630920407421098,
         0.24302232388731304, 0.22396294601870886, 0.20581705524736288, 0.18850106219379073, 0.1719423787555041,
         0.15607751257404356, 0.14085066964413634, 0.1262125008964556, 0.11211917907276119, 0.09853160263657085]

peaks = [0.43895537441354293,
         0.399092299516324,
         0.366053375598549,
         0.33784013584038697,
         0.31322089574365175,
         0.291382605707376,
         0.2717602793954051,
         0.25394534660217655,
         0.2376327497702539,
         0.22258869801692865,
         0.20863004858524786,
         0.19561066648125752,
         0.1834120783796856,
         0.17193689702539297,
         0.16110411139056832,
         0.15084558892382105,
         0.14110349552599757,
         0.13182832590084642,
         0.122977353808948,
         0.11451347625151148,
         0.10640423946795646,
         0.09862112466747279,
         0.09113890735187424,
         0.08393518546553809,
         0.07698997247883918]

# print()
# %%
peaks = [0.8404011571463119,
         0.7945252162031439,
         0.7565027674486093,
         0.7240338993470374,
         0.6957011425876662,
         0.670568808802902,
         0.6479866902539299,
         0.6274845888270858,
         0.6087114326964933,
         0.59139816643289,
         0.5753340223941958,
         0.5603508228778717,
         0.546312224280532,
         0.5331061496493349,
         0.5206393684159613,
         0.5088334710519952,
         0.497621899879081,
         0.4869476823216445,
         0.4767616474002418,
         0.46702109557436516,
         0.4576886778234583,
         0.44873157363966143,
         0.4401207537122128,
         0.431830436904474,
         0.4238376219588875]


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
    d = np.array([0.03532775, 0.04047159, 0.0469795, 0.05750226, 0.06787093,
                  0.07359354, 0.0765702, 0.07603304, 0.07444074, 0.07075915,
                  0.06562241, 0.05913068, 0.0520513, 0.04415979, 0.03651317,
                  0.02907329, 0.02347645, 0.01813943, 0.01403902, 0.01077079,
                  0.00842213, 0.00647535, 0.00518504, 0.00415951, 0.00323293])
    d /= np.sum(d)
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

    b /= np.sum(b)
    return b


def solver(C0, TIME, STEP, K, indx):
    prev = C0
    for t in np.arange(0, TIME, STEP):
        C_new = model(prev, STEP, K, indx)
        prev = C_new
    return prev


def count(state, N, K, indx):
    C = np.zeros((N + 1, N + 1, 2, state.shape[1]), dtype=np.float128)
    C[0, 6, 0, :] = state[0, :]  # + state[7, :]
    C[1, 5, 0, :] = state[1, :]  # + state[8, :]
    C[2, 4, 0, :] = state[2, :]  # + state[9, :]
    C[3, 3, 0, :] = state[3, :]  # + state[10, :]
    C[4, 2, 0, :] = state[4, :]  # + state[11, :]
    C[5, 1, 0, :] = state[5, :]  # + state[12, :]
    C[6, 0, 0, :] = state[6, :]  # + state[13, :]

    # C[:,:] /= np.sum(C, axis=(0,1))
    # c1 = solver(C, T, STEP, K, indx)

    # C = np.zeros((N + 1, N + 1, state.shape[1]), dtype=np.float128)
    C[0, 6, 1, :] = state[7, :]  # + state[7, :]
    C[1, 5, 1, :] = state[8, :]  # + state[8, :]
    C[2, 4, 1, :] = state[9, :]  # + state[9, :]
    C[3, 3, 1, :] = state[10, :]  # + state[10, :]
    C[4, 2, 1, :] = state[11, :]  # + state[11, :]
    C[5, 1, 1, :] = state[12, :]  # + state[12, :]
    C[6, 0, 1, :] = state[13, :]  # + state[13, :]

    c = solver(C, T, STEP, K, indx)
    cc = [np.zeros((N + 1, state.shape[1])), np.zeros((N + 1, state.shape[1]))]
    for j in range(N + 1):
        cc[0][j, :] = c[j, N - j, 0, :]
        cc[1][j, :] = c[j, N - j, 1, :]
    for i in range(cc[0].shape[1]):
        cc[0][:, i] = smooth_profile(cc[0][:, i])
        cc[1][:, i] = smooth_profile(cc[1][:, i])
        # cc[0][:,i] /= np.sum(cc[0][:,i])
        # cc[1][:,i] /= np.sum(cc[1][:,i])

    c = c[:, :, 0, :] + c[:, :, 1, :]

    data = np.zeros((N + 1, state.shape[1]))
    for j in range(N + 1):
        data[j, :] = c[j, N - j, :]
    for i in range(data.shape[1]):
        data[:, i] = smooth_profile(data[:, i])

    # data /= np.sum(data, axis=0)

    delta = abs(data - get_b()[:, None])
    chi2 = np.sum(delta ** 2 / data, axis=0)
    delta = np.sum(delta ** 2 / (get_b() + 0.0001)[:, None] / (get_b() + 0.0001)[:, None], axis=0)

    return state, (data, cc), chi2, delta


def get_K_matrix(N):
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
    init_state = 14
    thread = 1

    if len(sys.argv) != 2:
        thread = 1
    else:
        thread = int(sys.argv[1])

    STEP = 0.1
    T = 300

    batch_size = 300
    best = 5
    multy = 50
    K, indx = get_K_matrix(N)
    batch = np.zeros((init_state, 1))
    c0 = np.array(
        [0.710540, 0.006813, 0.612019, 0.424776, 0.012973, 0.058014, 0.509011, 0.959315, 0.312849, 0.089476, 0.235293,
         0.015592, 0.061242, 0.659040
         ])
    # batch[:, 0] = c0
    # s0 = count(batch, N=N, K=K, indx=indx)
    # print(s0)
    res = {}
    # res = {s0[2][0]: s0[0].reshape(init_state)}
    print(res)
    min_data = None
    min_chi2 = None
    min_delta = None
    proc = 10
    if 1 == 1:
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
            dd = data[1]
            data = data[0]

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
            # print(q[0][1])
            if True:
                tt = ",".join(itertools.repeat("{:4f}", init_state)).format(*q[0][1])
                text = """
                    delta:      {:.4f}
                    chi2:       {:.4f}
                    state:      [{}]
                """.format(min_delta, min_chi2, tt)
                tqdm.write(text)

                fig, axs = plt.subplots(1, 1, figsize=(10, 10))
                axs.bar(range(0, N + 1), min_data,
                        label=tt)
                axs.bar(range(0, N + 1), get_b(),
                        label='experiment', alpha=0.5)

                axs.scatter(range(0, N + 1), dd[0][:, tmp[0]],
                            label='c1')
                axs.scatter(range(0, N + 1), dd[1][:, tmp[0]],
                            label='c2')
                axs.set(xlabel='i', ylabel='C')
                axs.legend(ncol=5)
                plt.savefig(
                    os.path.join('plots',str(thread) + 'chi2: {:.3f}, delta:{:.3f}'.format(min_chi2,
                                                                                           min_delta) + '_' + tt + '.png'))
                plt.clf()
                plt.close(fig)
                # break
                # 0.677,0.084,0.426,0.926,0.860,0.396,0.108
