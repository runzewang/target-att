import theano
import theano.tensor as tensor
class AttentionLayer(object):
    def __init__(self,tpamars,prefix,x):
        self.pctx = tensor.dot(x,tpamars[self._p(prefix,'U_att')])+tpamars[self._p(prefix,'b_att')]
    def _p(self,pp,name):
        return '%s_%s' % (pp, name)
    def AttentionProc(self,tpamars,prefix,h,x,mask):
        pstate = tensor.dot(h,tpamars[self._p(prefix,'W_att')])
        zero_mat = tensor.alloc(0.,pstate.shape[0],self.pctx.shape[0],self.pctx.shape[1],self.pctx.shape[2])
        pctx_f = zero_mat+self.pctx[None,:,:,:]
        pctx_ = pctx_f + pstate[:,None,:,:]
        pctx_ = tensor.tanh(pctx_)

        alpha = tensor.dot(pctx_,tpamars[self._p(prefix,'V_att')])
        alpha = alpha.reshape([alpha.shape[0], alpha.shape[1],alpha.shape[2]])
        alpha_max = tensor.max(alpha,axis=1)
        alpha = alpha - alpha_max[:,None,:]
        alpha = tensor.exp(alpha)
        if mask:
            alpha = alpha*mask[None,:,:]
        alpha = alpha/alpha.sum(1,keepdims=True)
        one_mat = tensor.alloc(1.,alpha.shape[0],alpha.shape[1],alpha.shape[2],x.shape[2])
        x_ = one_mat*x[None,:,:,:]
        alpha_ = one_mat*alpha[:,:,:,None]
        ctx = (x_*alpha_).sum(1)
        return ctx,alpha
    def AttentionProcQA(self,tpamars,prefix,h,x):
        pstate = tensor.dot(h,tpamars[self._p(prefix,'W_att')])
        pctx_ = self.pctx+pstate[None,None,:,:]
        pctx_ = tensor.tanh(pctx_)

        alpha = tensor.dot(pctx_,tpamars[self._p(prefix,'V_att')])
        alpha = alpha.reshape([alpha.shape[0], alpha.shape[1],alpha.shape[2]])
        alpha_max = tensor.max(alpha,axis=1)
        alpha = alpha - alpha_max[:,None,:]
        alpha = tensor.exp(alpha)
        alpha_sum = tensor.sum(alpha,axis=1)[:,None,:]
        alpha = alpha/(tensor.sum(alpha,axis=1)[:,None,:])

        return alpha,alpha_sum
    def AttentionProcSlice(self,h,ctx,alpha,x,pctx,mask,W_att,V_att):
        # pstate = tensor.dot(h,tpamars[self._p(prefix,'W_att')])
        pstate = tensor.dot(h,W_att)
        pctx_ = pctx + pstate[None,:,:]
        pctx_ = tensor.tanh(pctx_)

        # alpha = tensor.dot(pctx_,tpamars[self._p(prefix,'V_att')])
        alpha = tensor.dot(pctx_,V_att)
        alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])
        alpha = tensor.exp(alpha)
        if mask:
            alpha = alpha*mask
        alpha = alpha/alpha.sum(0,keepdims=True)
        ctx = (x*alpha[:,:,None]).sum(0)
        return ctx,alpha
    def AttentionProcScan(self,tpamars,prefix,h,x,mask):
        seqs=[h]
        init_ctx = tensor.alloc(0.,h.shape[1],h.shape[2]).astype('float32')
        init_alpha = tensor.alloc(0,x.shape[0],x.shape[1]).astype('float32')
        non_sequence =[x,self.pctx,mask,tpamars[self._p(prefix,'W_att')],tpamars[self._p(prefix,'V_att')]]
        output,update = theano.scan(self.AttentionProcSlice,
                                    sequences=seqs,
                                    outputs_info=[init_ctx,init_alpha],
                                    non_sequences=non_sequence)
        return output[0],output[1]
 