import matplotlib.pyplot as plt
import numpy as np
import sys
import json

BIN_NUM = 200
RANGE = (-7, 7)

if __name__ == '__main__':
    settings = json.load(open(sys.argv[1]))
    ofname = sys.argv[2] if len(sys.argv) > 2 else None
    values_list = []
    labels_list = []
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for target_settings in settings['target']:
        values = np.load(target_settings['layer_weight'])
        values = values.reshape(values.size, 1)
        print('min={}, max={}'.format(np.min(values), np.max(values)))
        print(values.shape)
        values_list.append(values)
        labels_list.append(target_settings['label'])

    if settings['draw_type'] == 'overlay':
        for i, values in enumerate(values_list):
            ax.hist(values, bins=BIN_NUM, alpha=0.5, range=RANGE, label=labels_list[i])
    elif settings['draw_type'] == 'sidebyside':
        ax.hist(values_list, bins=BIN_NUM, alpha=0.5, range=RANGE, label=labels_list)
    ax.set_xlim(RANGE[0], RANGE[1])
    ax.legend()
    fig.show()

    if ofname != None:
        plt.savefig(sys.argv[2])
    else:
        plt.show()
