import theano
import theano.tensor as tensor
class AttentionLayer:
    def __init__(self,tpamars,prefix,x):
        self.pctx = tensor.dot(x,tpamars[self._p(prefix,'U_att')])+tpamars[self._p(prefix,'b_att')]
    def _p(self,pp,name):
        return '%s_%s' % (pp, name)
    def AttentionProc(self,tpamars,prefix,h,x,mask):
        pstate = tensor.dot(h,tpamars[self._p(prefix,'W_att')])
        pctx_ = self.pctx + pstate[None,:,:]
        pctx_ = tensor.tanh(pctx_)

        alpha = tensor.dot(pctx_,tpamars[self._p(prefix,'V_att')])
        alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])
        alpha_max = tensor.max(alpha,axis=0)
        alpha = alpha - alpha_max[None,:]
        alpha = tensor.exp(alpha)
        if mask:
            alpha = alpha*mask
        alpha = alpha/alpha.sum(0,keepdims=True)
        ctx = (x*alpha[:,:,None]).sum(0)
        return ctx,alpha
