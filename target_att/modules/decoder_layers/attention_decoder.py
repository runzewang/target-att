import theano
import theano.tensor as tensor
from modules.decoder_layers.attention_layer_single import AttentionLayer
from modules.encoder_layers.LSTM import LSTM
class Decoder(object):
    def __init__(self,tparams,prefix,h,context,mask):
        AttentionLayerIns = AttentionLayer(tparams,prefix,context)
        self.ctx,self.alphe  = AttentionLayerIns.AttentionProc(tparams,prefix,h,context,mask)
        self.n_samples = context.shape[1]
    def _p(self,pp,name):
        return '%s_%s' % (pp, name)
    def _slice(self,_x,n,dim):
        if _x.ndim == 3:
            return  _x[:,:,n*dim:(n+1)*dim]
        return _x[:,n*dim:(n+1)*dim]
    def decoder(self,tparams,options,prefix,pre_h,pre_c,state_below,mask):
        if mask is None:
            mask = tensor.alloc(1.,state_below.shape[0])
        LstmIns = LSTM(tparams,options,prefix,state_below,mask)
        state = tensor.dot(self.ctx,tparams[self._p(prefix,'Wc')])+LstmIns.state_below_
        # state = tensor.dot(self.ctx,tparams[self._p(prefix,'Wc')])
        h_state,c = LstmIns.lstm_layer(LstmIns.mask,state,pre_h,pre_c)
        return h_state,c,self.alphe