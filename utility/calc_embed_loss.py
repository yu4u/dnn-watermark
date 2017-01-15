import numpy as np
import sys

# caluculate cross-entropy (Note: p(0) = 0 and p(1) = 1 => H(p, q) = -sigma(p(x)*log(q(x)) = -log(q(x)) )
def calculate_loss(values):
    return -np.log(values).sum() / values.size

if __name__ == '__main__':
    pred_bparam_fname = sys.argv[1]
    values = np.load(pred_bparam_fname)
    print(values)
    print(calculate_loss(values))
