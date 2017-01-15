import pandas as pd
import sys

def dump_h5_keys(h5fname):
    store = pd.HDFStore(h5fname)
    print('<{}>'.format(h5fname))
    print('\n'.join(store.keys()))
    print()

if __name__ == '__main__':
    dump_h5_keys(sys.argv[1])
