import numpy as np
import sys
import re
import os

def get_model(N, k, nb_classes):
    model_script_path = os.path.join(os.path.dirname(__file__), '..')
    model_script_path = os.path.abspath(model_script_path)
    sys.path.append(model_script_path)
    import wide_residual_network as wrn

    init_shape = (32, 32, 3) 
    model = wrn.create_wide_residual_network(init_shape, nb_classes=nb_classes, N=N, k=k, dropout=0.00)
    return model

def get_param_from_name(name):
    match = re.search('N(\d+)K(\d+)[^_]+_TBLK(\d+)_layer(\d+)', name)
    
    if match:
        N = int(match.group(1))
        k = int(match.group(2))
        target_blk_id = int(match.group(3))
        target_layer_id = int(match.group(4))
        return (N, k, target_blk_id, target_layer_id)
    return None

def get_layer_weights_and_predicted(weight_fname, wparam_fname, nb_classes):
    N, k, target_blk_id, target_layer_id = get_param_from_name(wparam_fname)

    # create model and load weights
    model = get_model(N, k, nb_classes)
    model.load_weights(weight_fname)
    w = np.load(wparam_fname)

    # get signature from model weight and matrix
    target_layer = model.layers[target_layer_id]
    layer_weights = target_layer.get_weights()
    weight = (np.array(layer_weights[0])).mean(axis=3)
    pred_bparam = np.dot(weight.reshape(1, weight.size), w)  # dot product
    pred_bparam = 1 / (1 + np.exp(-pred_bparam))  # apply sigmoid
    return layer_weights[0], pred_bparam

if __name__ == '__main__':
    weight_fname = sys.argv[1]
    wparam_fname = sys.argv[2]
    oprefix = sys.argv[3]
    nb_classes = int(sys.argv[4]) if len(sys.argv) > 4 else 10 # caltech: 102

    # get file and parameters
    layer_weights, pred_bparam = get_layer_weights_and_predicted(weight_fname, wparam_fname, nb_classes)
    np.save('{}_predict_bparam.npy'.format(oprefix), pred_bparam)
    np.save('{}_layer_weight.npy'.format(oprefix), layer_weights)