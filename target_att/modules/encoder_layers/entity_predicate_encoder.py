from modules.encoder_layers.LSTM import LSTM
import theano
import theano.tensor as tensor
class EntityPredicateEncoder:
    def _p(self, pp, name):
        return '%s_%s' % (pp, name)
    def encoder(self,tparams,options,prefix,state_below,mask):
        encoder_layer = LSTM(tparams,options,prefix,state_below,mask)
        nsteps = state_below.shape[0]
        if state_below.ndim == 3:
            n_samples = state_below.shape[1]
        else:
            n_samples = 1
        if mask is None:
            mask = tensor.alloc(1.,state_below.shape[0],state_below.shape[1])
        seqs = [mask,encoder_layer.state_below_]
        init_states = tensor.alloc(0., n_samples, encoder_layer.dim)
        init_c = tensor.alloc(0., n_samples, encoder_layer.dim)
        rval, updates = theano.scan(encoder_layer.lstm_layer,
                            sequences=seqs,
                            outputs_info=[init_states,init_c],
                            n_steps=nsteps
                            )

        return rval[0]

