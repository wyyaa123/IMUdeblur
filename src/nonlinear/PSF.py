import numpy as np
import math
from operator import mod

class PSFFunction():
    def __init__(self, _len, _ang):
        """
        :param _len: 像素点位移大小
        :param _ang: 像素点偏移角度大小, 单位为 °
        """
        self.p2 = _len
        self.p3 = _ang

        self.len = max(1, self.p2)
        self.half = (self.len - 1) / 2 # rotate half length around center
        self.phi = mod(self.p3, 180) / 180 * math.pi

        self.cosphi = math.cos(self.phi)
        self.sinphi = math.sin(self.phi)
        self.xsign = np.sign(self.cosphi)
        self.linewdt = 1
        self.eps = 2.2204e-16 #matlab默认的最小浮点数精度

    def round_toward_zero(self, x):
        return int(abs(x)) * (1 if x > 0 else -1)

    # define mesh for the half matrix, eps takes care of the right size
    # for 0 & 90 rotation
    def calculate_mesh(self):
        sx = self.round_toward_zero(self.half * self.cosphi + self.linewdt * self.xsign - self.len * self.eps)
        sy = self.round_toward_zero(self.half * self.sinphi + self.linewdt - self.len * self.eps)
        m = 0
        n = 0
        a = []
        b = []
        if sx >= 0:
            while m <= sx:
                a.append(m)
                m = m + int(self.xsign)
        else:
            while m >= sx:
                a.append(m)
                m = m + int(self.xsign)
        while n <= sy:
            b.append(n)
            n = n + 1
        [self.x, self.y] = np.meshgrid(a, b)

    def calculate_h(self):
        self.calculate_mesh()
        # define shortest distance from a pixel to the rotated line
        self.dist2line = (self.y * self.cosphi - self.x * self.sinphi)

        self.rad = np.sqrt(self.x * self.x + self.y * self.y)

        # find points beyond the line's end-point but within the line width
        # self.lastpix = np.where(self.rad >= self.half and abs(self.dist2line) <= self.linewdt) #error
        self.ind = []
        for i in range(0, self.dist2line.shape[0]):
            for j in range(0, self.dist2line.shape[1]):
                if self.rad[i][j] >= self.half and abs(self.dist2line[i][j]) <= self.linewdt:
                    self.ind.append([i, j])

        # distance to the line's end-point parallel to the line
        self.x2lastpix = []
        for i in range(0, len(self.ind)):
            self.x2lastpix.append(self.half - abs((self.x[self.ind[i][0]][self.ind[i][1]] +
                                                   self.dist2line[self.ind[i][0]][self.ind[i][1]] *
                                                   self.sinphi) / self.cosphi))

        for i in range(0, len(self.ind)):
            self.dist2line[self.ind[i][0]][self.ind[i][1]] = math.sqrt(self.dist2line[self.ind[i][0]][self.ind[i][1]] *
                                                                       self.dist2line[self.ind[i][0]][self.ind[i][1]] +
                                                                       self.x2lastpix[i] * self.x2lastpix[i])

        self.dist2line = self.linewdt + self.eps - abs(self.dist2line)

        for i in range(0, self.dist2line.shape[0]):
            for j in range(0, self.dist2line.shape[1]):
                if self.dist2line[i][j] < 0:
                    self.dist2line[i][j] = 0

        # unfold half-matrix to the full size
        self.h = np.rot90(np.rot90(self.dist2line))

        self.hh = np.zeros((self.h.shape[0] * 2 - 1, self.h.shape[1] * 2 - 1))
        self.hh[0 : self.h.shape[0], 0 : self.h.shape[1]] = self.h
        self.hh[self.h.shape[0] - 1 : self.h.shape[0] * 2, self.h.shape[1] - 1 : self.h.shape[1] * 2] = self.dist2line

        self.hh = self.hh / (np.sum(self.hh) + self.eps * self.len * self.len)

        if self.cosphi > 0:
            self.hh = np.flipud(self.hh)

# test
if __name__ == '__main__':
    PSF = PSFFunction(33, 0)
    PSF.calculate_h()
    print(PSF.hh)
    np.savetxt('h.csv', PSF.hh, delimiter=',')


