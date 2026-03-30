import numpy as np
import scipy as sp
import numpy.matlib as npm
import numpy.linalg as npl
from Nargout import Nargout

class ReconstructionMethod(object):
    @staticmethod
    def Tikhonov_underdetermined(sysMatrix, measImage, alfa, iters):
        """from: Yang W, Peng L. Image reconstruction algorithms for electrical capacitance tomography[J]."""

        (m, n) = sysMatrix.shape
        vectorMeasImage = measImage

        eyeMatrix = np.eye(n, n)
        sys = np.dot((np.linalg.inv(np.dot(np.conj(sysMatrix.T), sysMatrix) + alfa * eyeMatrix)), np.conj(sysMatrix.T))

        xk0 = np.dot(sys, vectorMeasImage)
        xk0.imag = 0
        xk0[xk0 < 0] = 0
        xk = xk0
        for i in range(iters):
            xk = xk - np.dot(sys, (np.dot(sysMatrix, xk) - vectorMeasImage))
            xk.imag = 0
            xk[xk < 0] = 0

        error = []
        error.append(np.sum((sysMatrix.dot(xk) - vectorMeasImage) ** 2) / np.size(xk))
        return Nargout(xk, error)

