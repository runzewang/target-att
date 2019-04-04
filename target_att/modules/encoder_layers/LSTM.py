import theano
import theano.tensor as tensor
class LSTM:
    def __init__(self,tparams,options,prefix,state_below,mask):
        self.W = tparams[self._p(prefix,'W')]
        self.b = tparams[self._p(prefix,'b')]
        self.U = tparams[self._p(prefix,'U')]
        self.state_below_ = tensor.dot(state_below, self.W)+self.b
        self.dim = self.U.shape[0]
        self.mask = mask
    def _slice(self,_x,n,dim):
        if _x.ndim == 3:
            return  _x[:,:,n*dim:(n+1)*dim]
        return _x[:,n*dim:(n+1)*dim]
    def _p(self, pp, name):
        return '%s_%s' % (pp, name)
    def lstm_layer(self,m,x,pre_h,pre_c):
        preact = tensor.dot(pre_h,self.U)
        preact += x

        i = tensor.nnet.sigmoid(self._slice(preact, 0, self.dim))
        f = tensor.nnet.sigmoid(self._slice(preact, 1, self.dim))
        o = tensor.nnet.sigmoid(self._slice(preact, 2, self.dim))
        c = tensor.tanh(self._slice(preact, 3, self.dim))

        c = f*pre_c + i*c
        c = m[:,None]*c + (1. - m)[:,None]*pre_c

        h = o * tensor.tanh(c)
        h = m[:, None] * h + (1. - m)[:, None] * pre_h

        return h,c