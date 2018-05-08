import numpy as np
from enum import Enum

class RunMode(Enum):
    TRAIN=1
    VALIDATE=2
    TEST=3
    UNLABEL=4

def TopAccuracy(pred=None, truth=None, ratio=[1, 0.5, 0.2, 0.1]):
    if pred is None:
        print 'please provide a predicted contact matrix'
        sys.exit(-1)

    if truth is None:
        print 'please provide a true contact matrix'
        sys.exit(-1)

    assert pred.shape[0] == pred.shape[1]
    assert pred.shape == truth.shape

    pred_truth = np.dstack( (pred, truth) )

    M1s = np.ones_like(truth, dtype = np.int8)
    mask_LR = np.triu(M1s, 24)
    mask_MLR = np.triu(M1s, 12)
    mask_SMLR = np.triu(M1s, 6)
    mask_MR = mask_MLR - mask_LR
    mask_SR = mask_SMLR - mask_MLR

    seqLen = pred.shape[0]

    accs = []
    for mask in [ mask_LR, mask_MR, mask_MLR, mask_SR]:

        res = pred_truth[mask.nonzero()]
        res_sorted = res [ (-res[:,0]).argsort() ]

        for r in ratio:
    	    numTops = int(seqLen * r)
            numTops = min(numTops, res_sorted.shape[0] )
            topLabels = res_sorted[:numTops, 1]
            #numCorrects = ( (0<topLabels) & (topLabels<8) ).sum()
            numCorrects = np.sum(topLabels == 1.0)
            accuracy = numCorrects * 1./numTops
            accs.append(accuracy)

    return np.array(accs)

def conv_out_size_same(size, stride):
  return int(math.ceil(float(size) / float(stride)))
