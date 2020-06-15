print('loss params dim, weight')

def loss_params(dim,weight=25):
    def loss_function(actual,predicted):
        # import numpy as np
        from tensorflow.keras import backend as K
        Nsize = K.shape(predicted)[0]
        print(Nsize)
        mse = K.mean(K.square(predicted-actual),axis=-1)
        ell_inf = K.max(K.abs(K.sum(K.reshape(predicted,(Nsize,dim)),axis=1)-K.sum(K.reshape(actual,(Nsize,dim)),axis=1)))
        return mse+weight*ell_inf
    return loss_function

