import numpy
class InitLayersParamsDe:
    def __init__(self):
        pass
    def norm_weight(self,nin, nout=None, scale=0.01, ortho=True):
        if nout is None:
            nout = nin
        if nout == nin and ortho:
            W = self.ortho_weight(nin)
        else:
            W = scale * numpy.random.randn(nin, nout)
        return W.astype('float32')
    def _p(self,pp, name):
        return '%s_%s' % (pp, name)
    def ortho_weight(self,ndim):
        W = numpy.random.randn(ndim, ndim)
        u, s, v = numpy.linalg.svd(W)
        return u.astype('float32')

    def init_attention(self,options,params,prefix,nin=None,dim=None,dimctx=None):
        if nin is None:
            nin = options['dim']
        if dim is None:
            dim = options['dim']
        if dimctx is None:
            dimctx = options['dim']
        W_att = self.norm_weight(dim, dimctx)
        params[self._p(prefix, 'W_att')] = W_att

        # attention: context -> hidden
        U_att = self.norm_weight(dim,dimctx)
        params[self._p(prefix, 'U_att')] = U_att

        # attention: hidden bias
        b_att = numpy.zeros((dimctx,)).astype('float32')
        params[self._p(prefix, 'b_att')] = b_att

        # attention:
        V_att = self.norm_weight(dimctx, 1)
        params[self._p(prefix, 'V_att')] = V_att
        return params
    
    
    def init_decoder(self,options, params, prefix='lstm_en', nin=None, dim=None,dimctx=None):
        if nin is None:
            nin = options['dim_proj']
        if dim is None:
            dim = options['dim_proj']
        if dimctx is None:
            dimctx = options['dim']

        # embedding to gates transformation weights, biases
        W_ = numpy.concatenate([self.norm_weight(nin, dim),
                               self.norm_weight(nin, dim)], axis=1)
        Ws_ = numpy.concatenate([self.norm_weight(nin, dim),
                               self.norm_weight(nin, dim)], axis=1)
        W= numpy.concatenate([W_,Ws_], axis=1)
        params[self._p(prefix, 'W')] = W
        params[self._p(prefix, 'b')] = numpy.zeros((4 * dim,)).astype('float32')

        # recurrent transformation weights for gates
        U_ = numpy.concatenate([self.ortho_weight(dim),
                               self.ortho_weight(dim)], axis=1)
        Us_ = numpy.concatenate([self.ortho_weight(dim),
                               self.ortho_weight(dim)], axis=1)
        U = numpy.concatenate([U_,Us_], axis=1)
        params[self._p(prefix, 'U')] = U
        # maybe cause same problem
        Wc = self.norm_weight(nin, dimctx)
        params[self._p(prefix, 'Wc')] = Wc
        return params
