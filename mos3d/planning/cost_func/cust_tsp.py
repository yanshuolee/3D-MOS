from scipy import ndimage
import numpy as np

def calcDistVec(coord, offset):
    n = coord.shape[0]
    dist = np.zeros((n-offset+1, 1))
    temp = coord[:-offset, :] - coord[offset:, :]
    dist[1:] = np.sqrt(np.sum(temp**2, axis=1)).reshape((len(temp), 1))
    return dist

def solveTSP(vertex):
    N = len(vertex)
    itt = 0
    maxItt = min(20*N, 1e5) #min(3, 1e5) #
    noChange = 0
    order = np.arange(N)

    while (itt < maxItt) and (noChange < N/2): # 10
        dist = calcDistVec(vertex[order, :], 1)
        flip = np.mod(itt, N-3) + 2

        untie = dist[:N-flip] + dist[flip:]
        shuffledDist = calcDistVec(vertex[order, :], flip)
        connect = shuffledDist[:-1] + shuffledDist[1:]
        benefit = connect - untie

        localMin = ndimage.grey_erosion(benefit, footprint=np.ones((2*flip+1,1)))
        minimasInd = np.where(localMin == benefit)[0]
        reqFlips = minimasInd[(benefit[minimasInd] < -np.finfo(np.float64).eps).flatten()]

        prevOrd = order.copy() 
        for n in range(len(reqFlips)):
            order[reqFlips[n]:reqFlips[n]+flip] = order[reqFlips[n]:reqFlips[n]+flip][::-1]

        if (order == prevOrd).all():
            noChange = noChange + 1
        else:
            noChange = 0

        itt += 1

    return vertex[order, :], sum(dist)[0]
