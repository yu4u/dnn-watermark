import matplotlib.pyplot as plt
import pandas as pd
import sys
import json

def get_color_cycle_gen(colors):
    i = 0
    
    while True:
        yield colors[i]
        i = (i + 1) % len(colors)

if __name__ == '__main__':
    plot_setting_fname = sys.argv[1]
    ofname = sys.argv[2] if len(sys.argv) > 2 else None

    settings = json.load(open(plot_setting_fname))
    ax1 = ax2 = None
    color_cycle1 = get_color_cycle_gen(settings['color_order'])
    color_cycle2 = get_color_cycle_gen(settings['color_order'])
    lines = []

    for d in settings['target']:
        history_hdf5_fname = d['history']
        hdf5_path = d['path']
        label = d['label']
        store = pd.HDFStore(history_hdf5_fname)

        if not hdf5_path in store.keys():
            #print('"{}" is not included in hdf5'.format(hdf5_path), file=sys.stderr)
            break
        df = store[hdf5_path] # get data frame from store
        df_valacc = 1 - df['val_acc']
        df_loss = df['loss']
        index_offset = d['index_offset'] if 'index_offset' in d else 0
        df_valacc.index += index_offset
        df_loss.index += index_offset

        if ax1 == None:
            fig, ax1 = plt.subplots()
        lines.append(ax1.plot(df_valacc, linewidth=2, color=next(color_cycle1), label='{}'.format(label))[0])
        #ax1.plot(df['val_loss'], linewidth=2, label='val_loss({})'.format(label))

        if ax2 == None:
            ax2 = ax1.twinx()
        ax2.plot(df_loss, linewidth=2, linestyle='--', color=next(color_cycle2).format(label))[0]
        plt.xlabel('epoch')
        ax2.set_ylim(1e-2, 1e+1)
        ax2.set_yscale('log')

        store.close()

    ax1.set_ylabel('Test error')
    ax2.set_ylabel('Training loss')
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc=0)
    
    if ofname == None:
        plt.show()
    else:
        plt.savefig(ofname)
