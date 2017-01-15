from keras import backend as K
from keras.regularizers import Regularizer
import numpy as np

def random_index_generator(count):
    indices = np.arange(0, count)
    np.random.shuffle(indices)

    for idx in indices:
        yield idx

class WatermarkRegularizer(Regularizer):
    def __init__(self, k, b, wtype='random', randseed='none'):
        self.k = K.cast_to_floatx(k)
        self.uses_learning_phase = True
        self.wtype = wtype
        self.w = None
        self.p = None
        self.b = b

        if randseed == 'time':
            import time
            np.random.seed(int(time.time()))

    def set_param(self, p):
        if self.p is not None:
            raise Exception('Regularizers cannot be reused. '
                            'Instantiate one regularizer per layer.')
        self.p = p

        # make matrix
        p_shape = K.get_variable_shape(p)
        w_rows = np.prod(p_shape[0:3]) # todo: append theano pattern
        w_cols = self.b.shape[1]

        if self.wtype == 'random':
            self.w = np.random.randn(w_rows, w_cols)
        elif self.wtype == 'direct':
            self.w = np.zeros((w_rows, w_cols), dtype=None)
            rand_idx_gen = random_index_generator(w_rows)

            for col in range(w_cols):
                self.w[next(rand_idx_gen)][col] = 1.
        elif self.wtype == 'diff':
            self.w = np.zeros((w_rows, w_cols), dtype=None)
            rand_idx_gen = random_index_generator(w_rows)

            for col in range(w_cols):
                self.w[next(rand_idx_gen)][col] = 1.
                self.w[next(rand_idx_gen)][col] = -1.
        else:
            raise Exception('wtype="{}" is not supported'.format(self.wtype))

    def __call__(self, loss):
        if self.p is None:
            raise Exception('Need to call `set_param` on '
                            'WeightRegularizer instance '
                            'before calling the instance. ')
        regularized_loss = loss
        x = K.mean(self.p, axis=3)
        y = K.reshape(x, (1, K.count_params(x)))
        z = K.variable(value=self.w)
        regularized_loss += self.k * K.sum(K.binary_crossentropy(K.sigmoid(K.dot(y, z)), K.cast_to_floatx(self.b)))
        return K.in_train_phase(regularized_loss, loss)

    def set_layer(self, layer):
        print('called WatermarkRegularizer.set_layer()')
        super(WatermarkRegularizer, self).set_layer(layer)

    def get_config(self):
        return {'name': self.__class__.__name__,
                'k': float(self.k)}

    def get_matrix(self):
        return self.w

    def get_signature(self):
        return self.b

    def get_encoded_code(self):
        # this function is not work if set_layer was not called.
        layer = self.layer
        weights = layer.get_weights()
        weight = (np.array(weights[0])).mean(axis=3)
        print(K.eval(K.sigmoid(K.dot(K.variable(value=weight.reshape(1, weight.size)), K.variable(value=self.w)))))
        return None # todo

def get_wmark_regularizers(model):
    ret = []

    for i, layer in enumerate(model.layers):
        for regularizer in layer.regularizers:
            if str(regularizer.__class__).find('WatermarkRegularizer') >= 0:
                ret.append((i, regularizer))
    return ret

def show_encoded_wmark(model):
    for i, layer in enumerate(model.layers):
        for regularizer in layer.regularizers:
            if str(regularizer.__class__).find('WatermarkRegularizer') >= 0:
                print('<watermark code: layer_index={}, class={}>'.format(i, layer.__class__))
                weights = layer.get_weights()
                weight = (np.array(weights[0])).mean(axis=3)
                print(K.eval(K.sigmoid(K.dot(K.variable(value=weight.reshape(1, weight.size)), K.variable(value=regularizer.w)))))
                print(K.eval(K.sigmoid(K.dot(K.variable(value=weight.reshape(1, weight.size)), K.variable(value=regularizer.w)))) > 0.5)
