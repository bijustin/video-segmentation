import numpy as np
import cv2

# Code modified from: https://github.com/dganguli/robust-pca
# Example of use:
#
# rpca = R_pca(img_frames)
# L, S = rpca.fit(max_iter=10000, iter_print=100)


import numpy as np

class R_pca:

    def __init__(self, D, mu=None, lmbda=None):
        self.D = D
        self.S = np.zeros(self.D.shape)
        self.Y = np.zeros(self.D.shape)

        if mu:
            self.mu = mu
        else:
            self.mu = np.prod(self.D.shape) / (4 * self.norm_p(self.D, 2))

        self.mu_inv = 1 / self.mu

        if lmbda:
            self.lmbda = lmbda
        else:
            self.lmbda = 1 / np.sqrt(np.max(self.D.shape))

    @staticmethod
    def norm_p(M, p):
        return np.sum(np.power(M, p))

    @staticmethod
    def shrink(M, tau):
        return np.sign(M) * np.maximum((np.abs(M) - tau), np.zeros(M.shape))

    def svd_threshold(self, M, tau):
        U, S, V = np.linalg.svd(M, full_matrices=False)
        return np.dot(U, np.dot(np.diag(self.shrink(S, tau)), V))

    def fit(self, tol=None, max_iter=1000, iter_print=100, debug=False):
        iter = 0
        err = np.Inf
        Sk = self.S
        Yk = self.Y
        Lk = np.zeros(self.D.shape)

        if tol:
            _tol = tol
        else:
            _tol = 1E-7 * self.norm_p(np.abs(self.D), 2)

        while (err > _tol) and iter < max_iter:
            Lk = self.svd_threshold(
                self.D - Sk + self.mu_inv * Yk, self.mu_inv)
            Sk = self.shrink(
                self.D - Lk + (self.mu_inv * Yk), self.mu_inv * self.lmbda)
            Yk = Yk + self.mu * (self.D - Lk - Sk)
            err = self.norm_p(np.abs(self.D - Lk - Sk), 2)
            iter += 1
            if debug and ((iter % iter_print) == 0 or iter == 1 or iter > max_iter or err <= _tol):
                print('iteration: {0}, error: {1}'.format(iter, err))

        self.L = Lk
        self.S = Sk
        return Lk, Sk

def get_binary_map(current_frame, prev_frame=None, max_iter=1000, debug=False):
    current_frame = cv2.cvtColor(current_frame, cv2.COLOR_RGB2GRAY)
    if prev is not None:
        prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
        frames = np.stack((prev_frame, current_frame), axis=2)
    else:
        frames = np.expand_dims(current_frame,axis=2)
    frame_shape = frames.shape

    frames = frames.reshape(-1, frame_shape[2])
    rpca = R_pca(frames)
    L, S = rpca.fit(max_iter=max_iter, iter_print=100, debug=debug)
    S = S.reshape(frame_shape)
    S = cv2.normalize(S[:,:,0], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(np.uint8)
    _,S = cv2.threshold(S,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return S
