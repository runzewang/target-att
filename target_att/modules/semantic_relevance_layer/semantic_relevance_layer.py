import theano
import theano.tensor as tensor
class SemanticRelevanceLayer_untidy:
    def __init__(self):
        pass
    def CalculateCosine(self,options,ctx=None,proj_h=None,ctx_mask=None):
        r = options['r']
        fill_matrix = tensor.ones_like(ctx_mask)-ctx_mask
        norm_ctx = (ctx+fill_matrix[:,:,None]).norm(2, 2)*ctx_mask
        norm_proj = proj_h.norm(2,1)
        mul_cp = (ctx*proj_h[None,:,:]).sum(2)
        cos_cp = mul_cp/(norm_ctx*norm_proj[None,:]+fill_matrix)
        r_ = tensor.zeros_like(cos_cp)
        r_ = tensor.fill(r_,r)
        exp_cp = tensor.exp(cos_cp*r_)*ctx_mask
        p = exp_cp/(exp_cp.sum(0)[None,:]+tensor.min(fill_matrix, axis=0)[None,:])
        prob_max = p.argmax(0)
        return p,prob_max
    def CalculateCosineS(self,options,ctx=None,proj_h=None, h_mask=None):
        r = options['r']
        fill_matrix = tensor.ones_like(h_mask)-h_mask
        norm_ctx = ctx.norm(2, 2)
        norm_proj = (proj_h+fill_matrix[:,:,None]).norm(2, 2)*h_mask
        mul_cp = (ctx*proj_h).sum(2)
        cos_cp = mul_cp/(norm_ctx*norm_proj+fill_matrix)
        r_ = tensor.zeros_like(cos_cp)
        r_ = tensor.fill(r_,r)
        exp_cp = tensor.exp(cos_cp*r_)*h_mask

        p = exp_cp/(exp_cp.sum(0)[None,:]+tensor.min(fill_matrix,axis=0)[None, :])
        return p
    def CalculateCosine_webS(self,options,ctx=None,proj_h=None,mask_x=None):
        r = options['r']
        norm_ctx = ctx.norm(2,2)
        norm_proj = proj_h.norm(2,2)
        mul_cp = (ctx*proj_h).sum(2)
        cos_cp = mul_cp/(norm_ctx*norm_proj+0.0001)
        r_ = tensor.zeros_like(cos_cp)
        r_ = tensor.fill(r_,r)
        exp_cp = tensor.exp(cos_cp*r_)
        exp_cp_ = exp_cp*(mask_x.reshape([mask_x.shape[0], ctx.shape[0], ctx.shape[1]]).max(0))
        p = exp_cp_/(exp_cp_.sum(0)[None,:]+0.0001)
        return p
    def CalculateCosineCross(self,options,ctx_qu_rel,ctx_qu_wor,ctx_qu_cha,ctx_pr_rel,ctx_pr_wor,ctx_pr_cha,pr_alpha):
        p_rel = self.CalculateCosineS(options,ctx_qu_rel,ctx_pr_rel)
        p_wor = self.CalculateCosineS(options,ctx_qu_wor,ctx_pr_wor)
        p_cha = self.CalculateCosineS(options,ctx_qu_cha,ctx_pr_cha)
        p_ = tensor.concatenate([p_rel.dimshuffle(0,'x',1),p_wor.dimshuffle(0,'x',1)],axis=1)
        p = tensor.concatenate([p_,p_cha.dimshuffle(0,'x',1)],axis=1)*pr_alpha
        lats_p = p.sum(1)
        prob_max = lats_p.argmax(0)
        return  p_rel,p_wor,p_cha,lats_p, prob_max
    def CalculateCosineCross_relation_attention(self,options,ctx_qu,ctx_pr_rel,ctx_pr_wor,ctx_pr_cha, ctx_pr_mask):
        p_rel,p_max_rel = self.CalculateCosine(options,ctx_pr_rel,ctx_qu, ctx_pr_mask)
        p_wor,p_max_wor = self.CalculateCosine(options,ctx_pr_wor,ctx_qu, ctx_pr_mask)
        p_cha,p_max_cha = self.CalculateCosine(options,ctx_pr_cha,ctx_qu, ctx_pr_mask)

        p_ = tensor.concatenate([p_rel.dimshuffle(0,'x',1),p_wor.dimshuffle(0,'x',1)],axis=1)
        p = tensor.mean(tensor.concatenate([p_,p_cha.dimshuffle(0,'x',1)], axis=1),axis=1)
        prob_max = p.argmax(0)
        return p_rel,p_wor,p_cha, p, prob_max
    def CalculateCosineCross_withoutQA(self,options,ctx_qu_rel,ctx_qu_wor,ctx_qu_cha,ctx_pr_rel,ctx_pr_wor,ctx_pr_cha, ctx_pr_mask, pattern=''):
        p_rel = self.CalculateCosineS(options,ctx_qu_rel,ctx_pr_rel, ctx_pr_mask)
        p_wor = self.CalculateCosineS(options,ctx_qu_wor,ctx_pr_wor, ctx_pr_mask)
        p_cha = self.CalculateCosineS(options,ctx_qu_cha,ctx_pr_cha, ctx_pr_mask)
        pattern = 'RWC'
        if pattern == 'RWC':
            p_ = tensor.concatenate([p_rel.dimshuffle(0,'x',1),p_wor.dimshuffle(0,'x',1)],axis=1)
            p = tensor.mean(tensor.concatenate([p_,p_cha.dimshuffle(0,'x',1)], axis=1),axis=1)
            prob_max = p.argmax(0)
        if pattern == 'RW':
            p = tensor.mean(tensor.concatenate([p_rel.dimshuffle(0,'x',1),p_wor.dimshuffle(0,'x',1)], axis=1),axis=1)
            prob_max = p.argmax(0)
        if pattern == 'RC':
            p = tensor.mean(tensor.concatenate([p_rel.dimshuffle(0,'x',1),p_cha.dimshuffle(0,'x',1)],axis=1),axis=1)
            prob_max = p.argmax(0)
        if pattern == 'WC':
            p = tensor.mean(tensor.concatenate([p_wor.dimshuffle(0,'x',1),p_cha.dimshuffle(0,'x',1)],axis=1),axis=1)
            prob_max = p.argmax(0)
        if pattern == 'R':
            p = p_rel
            prob_max = p.argmax(0)
        if pattern == 'W':
            p = p_wor
            prob_max = p.argmax(0)
        if pattern == 'C':
            p = p_cha
            prob_max = p.argmax(0)

        return p_rel,p_wor,p_cha, p, prob_max
    def CalculateCosineCross_withoutQA_RWC(self,options,ctx_qu_rel,ctx_qu_wor,ctx_qu_cha,ctx_pr_rel,ctx_pr_wor,ctx_pr_cha, pattern=''):
        p_rel = self.CalculateCosineS(options,ctx_qu_rel,ctx_pr_rel)
        p_wor = self.CalculateCosineS(options,ctx_qu_wor,ctx_pr_wor)
        p_cha = self.CalculateCosineS(options,ctx_qu_cha,ctx_pr_cha)
        if pattern == 'RWC':
            p_ = tensor.concatenate([p_rel.dimshuffle(0,'x',1),p_wor.dimshuffle(0,'x',1)],axis=1)
            p = tensor.mean(tensor.concatenate([p_,p_cha.dimshuffle(0,'x',1)], axis=1),axis=1)
            prob_max = p.argmax(0)
        if pattern == 'RW':
            p = tensor.mean(tensor.concatenate([p_rel.dimshuffle(0,'x',1),p_wor.dimshuffle(0,'x',1)], axis=1),axis=1)
            prob_max = p.argmax(0)
        if pattern == 'RC':
            p = tensor.mean(tensor.concatenate([p_rel.dimshuffle(0,'x',1),p_cha.dimshuffle(0,'x',1)],axis=1),axis=1)
            prob_max = p.argmax(0)
        if pattern == 'WC':
            p = tensor.mean(tensor.concatenate([p_wor.dimshuffle(0,'x',1),p_cha.dimshuffle(0,'x',1)],axis=1),axis=1)
            prob_max = p.argmax(0)
        if pattern == 'R':
            p = p_rel
            prob_max = p.argmax(0)
        if pattern == 'W':
            p = p_wor
            prob_max = p.argmax(0)
        if pattern == 'C':
            p = p_cha
            prob_max = p.argmax(0)
        return p_rel,p_wor,p_cha, p, prob_max

    def CalculateCosineCross_withoutQA_RW(self,options,ctx_qu_rel,ctx_qu_wor,ctx_pr_rel,ctx_pr_wor):
        p_rel = self.CalculateCosineS(options,ctx_qu_rel,ctx_pr_rel)
        p_wor = self.CalculateCosineS(options,ctx_qu_wor,ctx_pr_wor)
        p = tensor.mean(tensor.concatenate([p_rel.dimshuffle(0,'x',1),p_wor.dimshuffle(0,'x',1)], axis=1),axis=1)
        prob_max = p.argmax(0)
        return p, prob_max
    def CalculateCosineCross_withoutQA_RC(self,options,ctx_qu_rel,ctx_qu_cha,ctx_pr_rel,ctx_pr_cha):
        p_rel = self.CalculateCosineS(options,ctx_qu_rel,ctx_pr_rel)
        p_cha = self.CalculateCosineS(options,ctx_qu_cha,ctx_pr_cha)
        p = tensor.mean(tensor.concatenate([p_rel.dimshuffle(0,'x',1),p_cha.dimshuffle(0,'x',1)],axis=1),axis=1)
        prob_max = p.argmax(0)
        return p, prob_max
    def CalculateCosineCross_withoutQA_WC(self,options,ctx_qu_wor,ctx_qu_cha,ctx_pr_wor,ctx_pr_cha):
        p_wor = self.CalculateCosineS(options,ctx_qu_wor,ctx_pr_wor)
        p_cha = self.CalculateCosineS(options,ctx_qu_cha,ctx_pr_cha)
        p = tensor.mean(tensor.concatenate([p_wor.dimshuffle(0,'x',1),p_cha.dimshuffle(0,'x',1)],axis=1),axis=1)
        prob_max = p.argmax(0)
        return p, prob_max
    def CalculateCosineCross_withoutQA_R(self,options,ctx_qu_rel,ctx_pr_rel):
        p_rel = self.CalculateCosineS(options,ctx_qu_rel,ctx_pr_rel)
        prob_max = p_rel.argmax(0)
        return p_rel, prob_max
    def CalculateCosineCross_withoutQA_W(self,options,ctx_qu_wor,ctx_pr_wor):
        p_wor = self.CalculateCosineS(options,ctx_qu_wor,ctx_pr_wor)
        prob_max = p_wor.argmax(0)
        return p_wor, prob_max
    def CalculateCosineCross_withoutQA_C(self,options,ctx_qu_cha,ctx_pr_cha):
        p_cha = self.CalculateCosineS(options,ctx_qu_cha,ctx_pr_cha)
        prob_max = p_cha.argmax(0)
        return p_cha, prob_max
    def CalculateCosineWeb_withoutQA(self,options,ctx_qu_rel,ctx_qu_wor,ctx_qu_cha,ctx_pr_rel,ctx_pr_wor,ctx_pr_cha,mask_cha,mask_wor,mask_rel):
        p_rel = self.CalculateCosine_webS(options,ctx_qu_rel,ctx_pr_rel,mask_rel)
        p_wor = self.CalculateCosine_webS(options,ctx_qu_wor,ctx_pr_wor,mask_wor)
        p_cha = self.CalculateCosine_webS(options,ctx_qu_cha,ctx_pr_cha,mask_cha)

        p_ = tensor.concatenate([p_rel.dimshuffle(0,'x',1),p_wor.dimshuffle(0,'x',1)],axis=1)
        p = tensor.mean(tensor.concatenate([p_,p_cha.dimshuffle(0,'x',1)],axis=1),axis=1)
        prob_max = p.argmax(0)
        return  p_rel,p_wor,p_cha, p, prob_max
