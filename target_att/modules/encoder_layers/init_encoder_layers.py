import numpy
import codecs
import json
class InitLayersParamsEn:
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

    def init_question_encoder_lstm(self,options,params,prefix,nin=None,dim=None):
        if options['lstm_layers'] == 1:
            if nin is None:
                nin = options['dim_proj']
            if dim is None:
                dim = options['dim_proj']
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
        if options['lstm_layers'] == 2:
            prefix1 = self._p(prefix,str(1))
            prefix2 = self._p(prefix,str(2))
            if nin is None:
                nin = options['dim_proj']
            if dim is None:
                dim = options['dim_proj']
            W_ = numpy.concatenate([self.norm_weight(nin, dim),
                                    self.norm_weight(nin, dim)], axis=1)
            Ws_ = numpy.concatenate([self.norm_weight(nin, dim),
                                    self.norm_weight(nin, dim)], axis=1)
            W= numpy.concatenate([W_,Ws_], axis=1)
            params[self._p(prefix1, 'W')] = W
            params[self._p(prefix1, 'b')] = numpy.zeros((4 * dim,)).astype('float32')

            # recurrent transformation weights for gates
            U_ = numpy.concatenate([self.ortho_weight(dim),
                                   self.ortho_weight(dim)], axis=1)
            Us_ = numpy.concatenate([self.ortho_weight(dim),
                                   self.ortho_weight(dim)], axis=1)
            U = numpy.concatenate([U_,Us_], axis=1)
            params[self._p(prefix1, 'U')] = U


            W_2 = numpy.concatenate([self.norm_weight(nin, dim),
                                    self.norm_weight(nin, dim)], axis=1)
            Ws_2 = numpy.concatenate([self.norm_weight(nin, dim),
                                    self.norm_weight(nin, dim)], axis=1)
            W2= numpy.concatenate([W_2,Ws_2], axis=1)
            params[self._p(prefix2, 'W')] = W2
            params[self._p(prefix2, 'b')] = numpy.zeros((4 * dim,)).astype('float32')

            # recurrent transformation weights for gates
            U_2 = numpy.concatenate([self.ortho_weight(dim),
                                   self.ortho_weight(dim)], axis=1)
            Us_2 = numpy.concatenate([self.ortho_weight(dim),
                                   self.ortho_weight(dim)], axis=1)
            U2 = numpy.concatenate([U_2,Us_2], axis=1)
            params[self._p(prefix2, 'U')] = U2
        return params
    def init_entity_encoder_lstm(self,options, params, prefix='lstm_en', nin=None, dim=None):
        if options['lstm_layers'] == 1:
            if nin is None:
                nin = options['dim_proj']
            if dim is None:
                dim = options['dim_proj']

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
            if options['lstm_end'] == 'joint':
                J =  self.norm_weight(2*dim, dim)
                params[self._p(prefix, 'joint')] = J
        if options['lstm_layers'] == 2:
            prefix1 = self._p(prefix,str(1))
            prefix2 = self._p(prefix,str(2))
            if nin is None:
                nin = options['dim_proj']
            if dim is None:
                dim = options['dim_proj']

            # embedding to gates transformation weights, biases
            W_ = numpy.concatenate([self.norm_weight(nin, dim),
                                   self.norm_weight(nin, dim)], axis=1)
            Ws_ = numpy.concatenate([self.norm_weight(nin, dim),
                                   self.norm_weight(nin, dim)], axis=1)
            W= numpy.concatenate([W_,Ws_], axis=1)
            params[self._p(prefix1, 'W')] = W
            params[self._p(prefix1, 'b')] = numpy.zeros((4 * dim,)).astype('float32')

            # recurrent transformation weights for gates
            U_ = numpy.concatenate([self.ortho_weight(dim),
                                   self.ortho_weight(dim)], axis=1)
            Us_ = numpy.concatenate([self.ortho_weight(dim),
                                   self.ortho_weight(dim)], axis=1)
            U = numpy.concatenate([U_,Us_], axis=1)
            params[self._p(prefix1, 'U')] = U
            if options['lstm_end'] == 'joint':
                J =  self.norm_weight(2*dim, dim)
                params[self._p(prefix1, 'joint')] = J


            W_2 = numpy.concatenate([self.norm_weight(nin, dim),
                                   self.norm_weight(nin, dim)], axis=1)
            Ws_2 = numpy.concatenate([self.norm_weight(nin, dim),
                                   self.norm_weight(nin, dim)], axis=1)
            W2= numpy.concatenate([W_2,Ws_2], axis=1)
            params[self._p(prefix2, 'W')] = W2
            params[self._p(prefix2, 'b')] = numpy.zeros((4 * dim,)).astype('float32')

            # recurrent transformation weights for gates
            U_2 = numpy.concatenate([self.ortho_weight(dim),
                                   self.ortho_weight(dim)], axis=1)
            Us_2= numpy.concatenate([self.ortho_weight(dim),
                                   self.ortho_weight(dim)], axis=1)
            U2 = numpy.concatenate([U_2,Us_2], axis=1)
            params[self._p(prefix2, 'U')] = U2
            if options['lstm_end'] == 'joint':
                J2 =  self.norm_weight(2*dim, dim)
                params[self._p(prefix2, 'joint')] = J2
        return params
    def init_predicate_encoder_lstm(self,options, params, prefix='lstm_pr', nin=None, dim=None):
        if options['lstm_layers'] == 1:
            if nin is None:
                nin = options['dim_proj']
            if dim is None:
                dim = options['dim_proj']

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
            if options['word'] ==  True:
                w_emb = self.norm_weight(options['word_dict_num'], dim)
                params[self._p(prefix, 'Wemb')] = w_emb
            if options['lstm_end'] == 'joint':
                J =  self.norm_weight(2*dim, dim)
                params[self._p(prefix, 'joint')] = J
            if options['lstm_end'] == 'attention':
                W_att = self.norm_weight(dim, dim)
                params[self._p(prefix, 'W_att')] = W_att

                # attention: context -> hidden
                U_att = self.norm_weight(dim,dim)
                params[self._p(prefix, 'U_att')] = U_att

                # attention: hidden bias
                b_att = numpy.zeros((dim,)).astype('float32')
                params[self._p(prefix, 'b_att')] = b_att

                # attention:
                V_att = self.norm_weight(dim, 1)
                params[self._p(prefix, 'V_att')] = V_att
        if options['lstm_layers'] == 2:
            prefix1 = self._p(prefix,str(1))
            prefix2 = self._p(prefix,str(2))
            if nin is None:
                nin = options['dim_proj']
            if dim is None:
                dim = options['dim_proj']

            # embedding to gates transformation weights, biases
            W_ = numpy.concatenate([self.norm_weight(nin, dim),
                                   self.norm_weight(nin, dim)], axis=1)
            Ws_ = numpy.concatenate([self.norm_weight(nin, dim),
                                   self.norm_weight(nin, dim)], axis=1)
            W= numpy.concatenate([W_,Ws_], axis=1)
            params[self._p(prefix1, 'W')] = W
            params[self._p(prefix1, 'b')] = numpy.zeros((4 * dim,)).astype('float32')

            # recurrent transformation weights for gates
            U_ = numpy.concatenate([self.ortho_weight(dim),
                                   self.ortho_weight(dim)], axis=1)
            Us_ = numpy.concatenate([self.ortho_weight(dim),
                                   self.ortho_weight(dim)], axis=1)
            U = numpy.concatenate([U_,Us_], axis=1)
            params[self._p(prefix1, 'U')] = U
            if options['word'] ==  True:
                w_emb = self.norm_weight(options['word_dict_num'], dim)
                params[self._p(prefix, 'Wemb')] = w_emb
            if options['lstm_end'] == 'joint':
                J =  self.norm_weight(2*dim, dim)
                params[self._p(prefix1, 'joint')] = J


            W_2 = numpy.concatenate([self.norm_weight(nin, dim),
                                   self.norm_weight(nin, dim)], axis=1)
            Ws_2 = numpy.concatenate([self.norm_weight(nin, dim),
                                   self.norm_weight(nin, dim)], axis=1)
            W2= numpy.concatenate([W_2,Ws_2], axis=1)
            params[self._p(prefix2, 'W')] = W2
            params[self._p(prefix2, 'b')] = numpy.zeros((4 * dim,)).astype('float32')

            # recurrent transformation weights for gates
            U_2 = numpy.concatenate([self.ortho_weight(dim),
                                   self.ortho_weight(dim)], axis=1)
            Us_2= numpy.concatenate([self.ortho_weight(dim),
                                   self.ortho_weight(dim)], axis=1)
            U2 = numpy.concatenate([U_2,Us_2], axis=1)
            params[self._p(prefix2, 'U')] = U2
            if options['lstm_end'] == 'joint':
                J2 =  self.norm_weight(2*dim, dim)
                params[self._p(prefix2, 'joint')] = J2
        return params
   
    def init_embed(self,options, params, prefix='cnn_en', num=None, nin=None):
        w_emb = self.norm_weight(num, nin)
        params[self._p(prefix, 'emb')] = w_emb
        return params

