import theano
import theano.tensor as tensor
class Criterions:
    def __init__(self):
        pass
    def NLLCriterions(self,prob,y,pro):
        # y = theano.printing.Print(pro+'y')(y)
        # prob = theano.printing.Print(pro+'prob')(prob)
        prob_ = prob[y, tensor.arange(y.shape[0])]
        # prob_ = theano.printing.Print(pro+'prob_')(prob_)
        return -tensor.mean(tensor.log(prob_))
    def errors(self,en_pred,pr_pred,t):
        return tensor.and_(tensor.eq(en_pred, t[0,:]),tensor.eq(pr_pred,t[1,:])).sum(0)
    def errors_notsum(self,en_pred,pr_pred,t):
        return tensor.and_(tensor.eq(en_pred,t[0,:]),tensor.eq(pr_pred,t[1,:]))
    def en_errors(self,en_pred,t):
        return tensor.eq(en_pred,t[0,:]).sum(0)
    def pr_errors(self,pr_pred,t):
        return tensor.eq(pr_pred,t[1,:]).sum(0)
    def en_errors_notsum(self,en_pred,t):
        return tensor.eq(en_pred,t[0,:])
    def pr_errors_notsum(self,pr_pred,t):
        return tensor.eq(pr_pred,t[1,:])
    def errors_num_en(self,en_pred,t):
        return tensor.neq(en_pred,t[0,:])
    def errors_num_pr(self,pr_pred,t):
        return tensor.neq(pr_pred,t[1,:])
