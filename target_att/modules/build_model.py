import theano
import theano.tensor as tensor
from modules.encoder_layers.question_encoder import QuestionEncoderLstm
from modules.encoder_layers.entity_predicate_encoder import EntityPredicateEncoder

from modules.decoder_layers.attention_decoder import Decoder

from modules.decoder_layers.attention_layer import AttentionLayer
from modules.semantic_relevance_layer.semantic_relevance_layer import SemanticRelevanceLayer_untidy
from modules.semantic_relevance_layer.criterion import Criterions
class MODEL:
    def __init__(self):
        pass
    def BuildTrainModel(self,tparams,options):
        Wemb = tensor.eye(options['dim_word'],dtype='float32')
        x = tensor.matrix('x', dtype='int64')
        x_mask = tensor.matrix('x_mask', dtype='float32')

        y = tensor.matrix('y', dtype='int64')
        y_mask = tensor.matrix('y_mask', dtype='float32')

        z_rel = tensor.tensor4('z_rel', dtype='int64')
        z_mask_rel = tensor.tensor4('z_mask_rel', dtype='float32')

        z_cha = tensor.tensor4('z_cha', dtype='int64')
        z_mask_cha = tensor.tensor4('z_mask_cha', dtype='float32')

        z_wor = tensor.tensor4('z_wor', dtype='int64')
        z_mask_wor = tensor.tensor4('z_mask_wor', dtype='float32')

        t = tensor.matrix('t',dtype='int64')

        xr = x[::-1]
        xr_mask = x_mask[::-1]

        yr = y[::-1]
        yr_mask = y_mask[::-1]


        n_timesteps_qu = x.shape[0]
        n_samples_qu = x.shape[1]

        n_timesteps_en = y.shape[0]
        n_samples_en = y.shape[1]

        n_timesteps_pr_cha = z_cha.shape[0]
        n_timesteps_pr_wor = z_wor.shape[0]
        n_timesteps_pr_rel = z_rel.shape[0]
        n_samples_pr = z_cha.shape[1]

        emb_qu = Wemb[x.flatten()]
        emb_qu = emb_qu.reshape([n_timesteps_qu,n_samples_qu,options['dim_word']])
        QuestionEncoderLstmIns = QuestionEncoderLstm()
        proj_qu = QuestionEncoderLstmIns.encoder(tparams,options,prefix='encoder_qu_first',state_below=emb_qu,mask=x_mask)

        emb_qur = Wemb[xr.flatten()]
        emb_qur = emb_qur.reshape([n_timesteps_qu,n_samples_qu,options['dim_word']])
        QuestionEncoderLstmInsr = QuestionEncoderLstm()
        proj_qur = QuestionEncoderLstmInsr.encoder(tparams,options,prefix='encoder_qur_first',state_below=emb_qur,mask=xr_mask)

        ctx_qu_first = tensor.concatenate([proj_qu, proj_qur[::-1]], axis=proj_qu.ndim-1)
        ctx_qu_firstr= ctx_qu_first[::-1]

        QuestionEncoderLstmIns_sec = QuestionEncoderLstm()
        proj_qu_sec = QuestionEncoderLstmIns_sec.encoder(tparams,options,prefix='encoder_qu_second',state_below=ctx_qu_first,mask=x_mask)
        QuestionEncoderLstmInsr_sec = QuestionEncoderLstm()
        proj_qur_sec = QuestionEncoderLstmInsr_sec.encoder(tparams,options,prefix='encoder_qur_second',state_below=ctx_qu_firstr,mask=xr_mask)
        ctx_qu_second = tensor.concatenate([proj_qu_sec, proj_qur_sec[::-1]], axis=proj_qu_sec.ndim-1)

        # ctx_qu = tensor.max(((ctx_qu_first+ctx_qu_second)*x_mask[:,:,None]),axis=0)

        emb_en = Wemb[y.flatten()]
        emb_en = emb_en.reshape([n_timesteps_en, n_samples_en, options['dim_word']])
        EntityPredicateEncoderInsEn = EntityPredicateEncoder()
        proj_en = EntityPredicateEncoderInsEn.encoder(tparams,options,prefix='encoder_en',state_below=emb_en,mask=y_mask)
        # ctx_en = proj_en.reshape([n_samples_en//n_samples_qu,n_samples_qu,options['dim']])

        embr_en = Wemb[yr.flatten()]
        embr_en = embr_en.reshape([n_timesteps_en, n_samples_en, options['dim_word']])
        EntityPredicateEncoderInsEnr = EntityPredicateEncoder()
        projr_en = EntityPredicateEncoderInsEnr.encoder(tparams,options,prefix='encoderr_en',state_below=embr_en,mask=yr_mask)
        # ctxr_en = projr_en.reshape([n_samples_en//n_samples_qu,n_samples_qu,options['dim']])

        chy_maks_utidy = tensor.max(y_mask,axis=0)
        fill_chy_maks_matrix = tensor.ones_like(chy_maks_utidy)-chy_maks_utidy
        ctx_en = (tensor.sum((tensor.concatenate([proj_en, projr_en[::-1]], axis=proj_en.ndim-1)*y_mask[:,:,None]),axis=0)/((tensor.sum(y_mask,axis=0)+fill_chy_maks_matrix)[:,None])).reshape([n_samples_en//n_samples_qu,n_samples_qu,options['dim_word']])
        # one timesteps for decoder
        SemanticRelevanceLayerIns = SemanticRelevanceLayer_untidy()
        y_mask_untidy = tensor.max(y_mask, axis=0).reshape([n_samples_en//n_samples_qu,n_samples_qu])


        h0 = tensor.alloc(0.,n_samples_qu, options['dim_word'])
        preh0 = tensor.alloc(0., n_samples_qu, options['dim_word'])
        prec0 = tensor.alloc(0., n_samples_qu, options['dim_word'])
        state_below = tensor.alloc(0.,n_samples_qu,options['dim_word'])
        mask_en = tensor.alloc(1.,n_samples_qu, 1)
        DecoderEn = Decoder(tparams,prefix='decoder_en', h=h0, context=ctx_qu_second,mask=x_mask)
        h_en,c_en,alphe_en = DecoderEn.decoder(tparams,options,prefix='decoder',pre_h=preh0,pre_c=prec0,state_below=state_below,mask=mask_en)
        p_en, p_max_en = SemanticRelevanceLayerIns.CalculateCosine(options,ctx_en,h_en, y_mask_untidy)

        ####character level relation embedding
        # ctx_pr_cha = tensor.alloc(0.,n_samples_pr,n_samples_qu,options['dim_word'])
        chz_mask_cha = (z_mask_cha[:,:,0,:].flatten()).reshape([n_timesteps_pr_cha, n_samples_pr*n_samples_qu])
        chz_maks_utidy = tensor.max(chz_mask_cha,axis=0)
        fill_chz_maks_matrix = tensor.ones_like(chz_maks_utidy)-chz_maks_utidy
        
        ctz_cha = z_cha[:,:,0,:]
        chz_mask_cha = (z_mask_cha[:,:,0,:].flatten()).reshape([n_timesteps_pr_cha, n_samples_pr*n_samples_qu])
        # chz_mask__cha =chz_mask.reshape([n_timesteps_pr, n_samples_pr*n_samples_qu])
        emb_pr_cha = tparams['character_emb'][ctz_cha.flatten()]
        emb_pr_cha = emb_pr_cha.reshape([n_timesteps_pr_cha, n_samples_pr*n_samples_qu, options['dim_word']])
        EntityPredicateEncoderInsPr = EntityPredicateEncoder()
        proj_pr_cha = EntityPredicateEncoderInsPr.encoder(tparams,options,prefix='encoder_pr_cha',state_below=emb_pr_cha,mask=chz_mask_cha)

        ctzr_cha = ctz_cha[::-1]
        chzr_mask_cha = chz_mask_cha[::-1]
        embr_pr_cha = tparams['character_emb'][ctzr_cha.flatten()]
        embr_pr_cha = embr_pr_cha.reshape([n_timesteps_pr_cha, n_samples_pr*n_samples_qu, options['dim_word']])
        EntityPredicateEncoderInsPrr = EntityPredicateEncoder()
        projr_pr_cha = EntityPredicateEncoderInsPrr.encoder(tparams,options,prefix='encoderr_pr_cha',state_below=embr_pr_cha,mask=chzr_mask_cha)

        ctx_pr_cha = (tensor.sum((tensor.concatenate([proj_pr_cha, projr_pr_cha[::-1]], axis=proj_pr_cha.ndim-1)*chz_mask_cha[:,:,None]),axis=0)/((tensor.sum(chz_mask_cha,axis=0)+fill_chz_maks_matrix)[:,None])).reshape([n_samples_pr,n_samples_qu,options['dim_word']])
        #####relation level relation embedding
        # ctx_pr_rel = tensor.alloc(0.,n_samples_pr,n_samples_qu,options['dim_word'])
        
        ctz_rel = z_rel[:,:,0,:]
        chz_mask_rel = (z_mask_rel[:,:,0,:].flatten()).reshape([n_timesteps_pr_rel, n_samples_pr*n_samples_qu])

        emb_pr_rel = tparams['relation_emb'][ctz_rel.flatten()]
        emb_pr_rel = emb_pr_rel.reshape([n_timesteps_pr_rel, n_samples_pr*n_samples_qu, options['dim_word']])
        EntityPredicateEncoderInsPr = EntityPredicateEncoder()
        proj_pr_rel = EntityPredicateEncoderInsPr.encoder(tparams,options,prefix='encoder_pr_rel',state_below=emb_pr_rel,mask=chz_mask_rel)

        ctzr_rel = ctz_rel[::-1]
        chzr_mask_rel = chz_mask_rel[::-1]
        embr_pr_rel = tparams['relation_emb'][ctzr_rel.flatten()]
        embr_pr_rel = embr_pr_rel.reshape([n_timesteps_pr_rel, n_samples_pr*n_samples_qu, options['dim_word']])
        EntityPredicateEncoderInsPrr = EntityPredicateEncoder()
        projr_pr_rel = EntityPredicateEncoderInsPrr.encoder(tparams,options,prefix='encoderr_pr_rel',state_below=embr_pr_rel,mask=chzr_mask_rel)

        ctx_pr_rel = (tensor.sum((tensor.concatenate([proj_pr_rel, projr_pr_rel[::-1]], axis=proj_pr_rel.ndim-1)*chz_mask_rel[:,:,None]),axis=0)/((tensor.sum(chz_mask_rel,axis=0)+fill_chz_maks_matrix)[:,None])).reshape([n_samples_pr,n_samples_qu,options['dim_word']])
        #### word level relation embedding
        # ctx_pr_wor = tensor.alloc(0.,n_samples_pr,n_samples_qu,options['dim_word'])

        ctz_wor = z_wor[:,:,0,:]
        chz_mask_wor = (z_mask_wor[:,:,0,:].flatten()).reshape([n_timesteps_pr_wor, n_samples_pr*n_samples_qu])

        emb_pr_wor = tparams['word_emb'][ctz_wor.flatten()]
        emb_pr_wor = emb_pr_wor.reshape([n_timesteps_pr_wor, n_samples_pr*n_samples_qu, options['dim_word']])
        EntityPredicateEncoderInsPr = EntityPredicateEncoder()
        proj_pr_wor = EntityPredicateEncoderInsPr.encoder(tparams,options,prefix='encoder_pr_wor',state_below=emb_pr_wor,mask=chz_mask_wor)

        ctzr_wor = ctz_wor[::-1]
        chzr_mask_wor = chz_mask_wor[::-1]
        embr_pr_wor = tparams['word_emb'][ctzr_wor.flatten()]
        embr_pr_wor = embr_pr_wor.reshape([n_timesteps_pr_wor, n_samples_pr*n_samples_qu, options['dim_word']])
        EntityPredicateEncoderInsPrr = EntityPredicateEncoder()
        projr_pr_wor = EntityPredicateEncoderInsPrr.encoder(tparams,options,prefix='encoderr_pr_wor',state_below=embr_pr_wor,mask=chzr_mask_wor)

        ctx_pr_wor = (tensor.sum((tensor.concatenate([proj_pr_wor, projr_pr_wor[::-1]], axis=proj_pr_wor.ndim-1)*chz_mask_wor[:,:,None]),axis=0)/((tensor.sum(chz_mask_wor,axis=0)+fill_chz_maks_matrix)[:,None])).reshape([n_samples_pr,n_samples_qu,options['dim_word']])


        ####target attention for relation score , word score, character score
        AttentionInsRel = AttentionLayer(tparams,prefix='att_rel',x=ctx_qu_second)
        ctx_qu_rel,ctx_qu_rel_alpha = AttentionInsRel.AttentionProc(tparams,prefix='att_rel',h=ctx_pr_rel,x=ctx_qu_second,mask = x_mask)
        AttentionInsWor = AttentionLayer(tparams,prefix='att_wor',x=ctx_qu_second)
        ctx_qu_wor,ctx_qu_wor_alpha = AttentionInsWor.AttentionProc(tparams,prefix='att_wor',h=ctx_pr_wor,x=ctx_qu_second,mask = x_mask)
        AttentionInsCha = AttentionLayer(tparams,prefix='att_cha',x=ctx_qu_second)
        ctx_qu_cha,ctx_qu_cha_alpha = AttentionInsCha.AttentionProc(tparams,prefix='att_cha',h=ctx_pr_cha,x=ctx_qu_second,mask = x_mask)

        ###calculate last relation score
        z_mask_untidy = tensor.max(chz_mask_cha, axis=0).reshape([n_samples_pr,n_samples_qu])
        p_rel,p_wor,p_cha,p_pr,p_max_pr = SemanticRelevanceLayerIns.CalculateCosineCross_withoutQA(options,ctx_qu_rel,ctx_qu_wor,ctx_qu_cha,ctx_pr_rel,ctx_pr_wor,ctx_pr_cha,z_mask_untidy,pattern='RWC')
        
        CriterionIns = Criterions()
        cost = CriterionIns.NLLCriterions(p_en,t[0,:],'en')+CriterionIns.NLLCriterions(p_pr, t[1,:], 'pr')
        errors = CriterionIns.errors(p_max_en,p_max_pr,t)

        return x, x_mask, y, y_mask,z_rel,z_mask_rel,z_wor,z_mask_wor,z_cha,z_mask_cha,t,cost,errors
    def BuildValidTestModel(self,tparams,options):
        Wemb = tensor.eye(options['dim_word'],dtype='float32')
        x = tensor.matrix('x', dtype='int64')
        x_mask = tensor.matrix('x_mask', dtype='float32')

        y = tensor.matrix('y', dtype='int64')
        y_mask = tensor.matrix('y_mask', dtype='float32')

        z_rel = tensor.tensor4('z_rel', dtype='int64')
        z_mask_rel = tensor.tensor4('z_mask_rel', dtype='float32')

        z_cha = tensor.tensor4('z_cha', dtype='int64')
        z_mask_cha = tensor.tensor4('z_mask_cha', dtype='float32')

        z_wor = tensor.tensor4('z_wor', dtype='int64')
        z_mask_wor = tensor.tensor4('z_mask_wor', dtype='float32')

        t = tensor.matrix('t',dtype='int64')

        xr = x[::-1]
        xr_mask = x_mask[::-1]

        yr = y[::-1]
        yr_mask = y_mask[::-1]


        n_timesteps_qu = x.shape[0]
        n_samples_qu = x.shape[1]

        n_timesteps_en = y.shape[0]
        n_samples_en = y.shape[1]

        n_timesteps_pr_cha = z_cha.shape[0]
        n_timesteps_pr_wor = z_wor.shape[0]
        n_timesteps_pr_rel = z_rel.shape[0]
        n_samples_pr = z_cha.shape[1]

        emb_qu = Wemb[x.flatten()]
        emb_qu = emb_qu.reshape([n_timesteps_qu,n_samples_qu,options['dim_word']])
        QuestionEncoderLstmIns = QuestionEncoderLstm()
        proj_qu = QuestionEncoderLstmIns.encoder(tparams,options,prefix='encoder_qu_first',state_below=emb_qu,mask=x_mask)

        emb_qur = Wemb[xr.flatten()]
        emb_qur = emb_qur.reshape([n_timesteps_qu,n_samples_qu,options['dim_word']])
        QuestionEncoderLstmInsr = QuestionEncoderLstm()
        proj_qur = QuestionEncoderLstmInsr.encoder(tparams,options,prefix='encoder_qur_first',state_below=emb_qur,mask=xr_mask)

        ctx_qu_first = tensor.concatenate([proj_qu, proj_qur[::-1]], axis=proj_qu.ndim-1)
        ctx_qu_firstr= ctx_qu_first[::-1]

        QuestionEncoderLstmIns_sec = QuestionEncoderLstm()
        proj_qu_sec = QuestionEncoderLstmIns_sec.encoder(tparams,options,prefix='encoder_qu_second',state_below=ctx_qu_first,mask=x_mask)
        QuestionEncoderLstmInsr_sec = QuestionEncoderLstm()
        proj_qur_sec = QuestionEncoderLstmInsr_sec.encoder(tparams,options,prefix='encoder_qur_second',state_below=ctx_qu_firstr,mask=xr_mask)
        ctx_qu_second = tensor.concatenate([proj_qu_sec, proj_qur_sec[::-1]], axis=proj_qu_sec.ndim-1)

        # ctx_qu = tensor.max(((ctx_qu_first+ctx_qu_second)*x_mask[:,:,None]),axis=0)

        emb_en = Wemb[y.flatten()]
        emb_en = emb_en.reshape([n_timesteps_en, n_samples_en, options['dim_word']])
        EntityPredicateEncoderInsEn = EntityPredicateEncoder()
        proj_en = EntityPredicateEncoderInsEn.encoder(tparams,options,prefix='encoder_en',state_below=emb_en,mask=y_mask)
        # ctx_en = proj_en.reshape([n_samples_en//n_samples_qu,n_samples_qu,options['dim']])

        embr_en = Wemb[yr.flatten()]
        embr_en = embr_en.reshape([n_timesteps_en, n_samples_en, options['dim_word']])
        EntityPredicateEncoderInsEnr = EntityPredicateEncoder()
        projr_en = EntityPredicateEncoderInsEnr.encoder(tparams,options,prefix='encoderr_en',state_below=embr_en,mask=yr_mask)
        # ctxr_en = projr_en.reshape([n_samples_en//n_samples_qu,n_samples_qu,options['dim']])
        chy_maks_utidy = tensor.max(y_mask,axis=0)
        fill_chy_maks_matrix = tensor.ones_like(chy_maks_utidy)-chy_maks_utidy
        ctx_en = (tensor.sum((tensor.concatenate([proj_en, projr_en[::-1]], axis=proj_en.ndim-1)*y_mask[:,:,None]),axis=0)/((tensor.sum(y_mask,axis=0)+fill_chy_maks_matrix)[:,None])).reshape([n_samples_en//n_samples_qu,n_samples_qu,options['dim_word']])
        # one timesteps for decoder
        SemanticRelevanceLayerIns = SemanticRelevanceLayer_untidy()
        y_mask_untidy = tensor.max(y_mask, axis=0).reshape([n_samples_en//n_samples_qu,n_samples_qu])
        
        h0 = tensor.alloc(0.,n_samples_qu, options['dim_word'])
        preh0 = tensor.alloc(0., n_samples_qu, options['dim_word'])
        prec0 = tensor.alloc(0., n_samples_qu, options['dim_word'])
        state_below = tensor.alloc(0.,n_samples_qu,options['dim_word'])
        mask_en = tensor.alloc(1.,n_samples_qu, 1)
        DecoderEn = Decoder(tparams,prefix='decoder_en', h=h0, context=ctx_qu_second,mask=x_mask)
        h_en,c_en,alphe_en = DecoderEn.decoder(tparams,options,prefix='decoder',pre_h=preh0,pre_c=prec0,state_below=state_below,mask=mask_en)
        # how to solve this problem
        # h_en_ = tensor.concatenate([h_en, h_en],axis=1)

        p_en,p_max_en = SemanticRelevanceLayerIns.CalculateCosine(options,ctx_en,h_en, y_mask_untidy)
    
        ####character level relation embedding
        # ctx_pr_cha = tensor.alloc(0.,n_samples_pr,n_samples_qu,options['dim_word'])
        chz_mask_cha = (z_mask_cha[:,:,p_max_en,tensor.arange(p_max_en.shape[0])].flatten()).reshape([n_timesteps_pr_cha, n_samples_pr*n_samples_qu])
        chz_maks_utidy = tensor.max(chz_mask_cha,axis=0)
        fill_chz_maks_matrix = tensor.ones_like(chz_maks_utidy)-chz_maks_utidy
        
        ctz_cha = z_cha[:,:,p_max_en,tensor.arange(p_max_en.shape[0])]
        chz_mask_cha = (z_mask_cha[:,:,p_max_en,tensor.arange(p_max_en.shape[0])].flatten()).reshape([n_timesteps_pr_cha, n_samples_pr*n_samples_qu])

        emb_pr_cha = tparams['character_emb'][ctz_cha.flatten()]
        emb_pr_cha = emb_pr_cha.reshape([n_timesteps_pr_cha, n_samples_pr*n_samples_qu, options['dim_word']])
        EntityPredicateEncoderInsPr = EntityPredicateEncoder()
        proj_pr_cha = EntityPredicateEncoderInsPr.encoder(tparams,options,prefix='encoder_pr_cha',state_below=emb_pr_cha,mask=chz_mask_cha)

        ctzr_cha = ctz_cha[::-1]
        chzr_mask_cha = chz_mask_cha[::-1]
        embr_pr_cha = tparams['character_emb'][ctzr_cha.flatten()]
        embr_pr_cha = embr_pr_cha.reshape([n_timesteps_pr_cha, n_samples_pr*n_samples_qu, options['dim_word']])
        EntityPredicateEncoderInsPrr = EntityPredicateEncoder()
        projr_pr_cha = EntityPredicateEncoderInsPrr.encoder(tparams,options,prefix='encoderr_pr_cha',state_below=embr_pr_cha,mask=chzr_mask_cha)


        ctx_pr_cha = (tensor.sum((tensor.concatenate([proj_pr_cha, projr_pr_cha[::-1]], axis=proj_pr_cha.ndim-1)*chz_mask_cha[:,:,None]),axis=0)/((tensor.sum(chz_mask_cha,axis=0)+fill_chz_maks_matrix)[:,None])).reshape([n_samples_pr,n_samples_qu,options['dim_word']])
        #####relation level relation embedding
        # ctx_pr_rel = tensor.alloc(0.,n_samples_pr,n_samples_qu,options['dim_word'])
        
        ctz_rel = z_rel[:,:,p_max_en,tensor.arange(p_max_en.shape[0])]
        chz_mask_rel = (z_mask_rel[:,:,p_max_en,tensor.arange(p_max_en.shape[0])].flatten()).reshape([n_timesteps_pr_rel, n_samples_pr*n_samples_qu])


        emb_pr_rel = tparams['relation_emb'][ctz_rel.flatten()]
        emb_pr_rel = emb_pr_rel.reshape([n_timesteps_pr_rel, n_samples_pr*n_samples_qu, options['dim_word']])
        EntityPredicateEncoderInsPr = EntityPredicateEncoder()
        proj_pr_rel = EntityPredicateEncoderInsPr.encoder(tparams,options,prefix='encoder_pr_rel',state_below=emb_pr_rel,mask=chz_mask_rel)

        ctzr_rel = ctz_rel[::-1]
        chzr_mask_rel = chz_mask_rel[::-1]
        embr_pr_rel = tparams['relation_emb'][ctzr_rel.flatten()]
        embr_pr_rel = embr_pr_rel.reshape([n_timesteps_pr_rel, n_samples_pr*n_samples_qu, options['dim_word']])
        EntityPredicateEncoderInsPrr = EntityPredicateEncoder()
        projr_pr_rel = EntityPredicateEncoderInsPrr.encoder(tparams,options,prefix='encoderr_pr_rel',state_below=embr_pr_rel,mask=chzr_mask_rel)

        ctx_pr_rel = (tensor.sum((tensor.concatenate([proj_pr_rel, projr_pr_rel[::-1]], axis=proj_pr_rel.ndim-1)*chz_mask_rel[:,:,None]),axis=0)/((tensor.sum(chz_mask_rel,axis=0)+fill_chz_maks_matrix)[:,None])).reshape([n_samples_pr,n_samples_qu,options['dim_word']])
        #### word level relation embedding
        # ctx_pr_wor = tensor.alloc(0.,n_samples_pr,n_samples_qu,options['dim_word'])
    

        ctz_wor = z_wor[:,:,p_max_en,tensor.arange(p_max_en.shape[0])]
        chz_mask_wor = (z_mask_wor[:,:,p_max_en,tensor.arange(p_max_en.shape[0])].flatten()).reshape([n_timesteps_pr_wor, n_samples_pr*n_samples_qu])

        emb_pr_wor = tparams['word_emb'][ctz_wor.flatten()]
        emb_pr_wor = emb_pr_wor.reshape([n_timesteps_pr_wor, n_samples_pr*n_samples_qu, options['dim_word']])
        EntityPredicateEncoderInsPr = EntityPredicateEncoder()
        proj_pr_wor = EntityPredicateEncoderInsPr.encoder(tparams,options,prefix='encoder_pr_wor',state_below=emb_pr_wor,mask=chz_mask_wor)

        ctzr_wor = ctz_wor[::-1]
        chzr_mask_wor = chz_mask_wor[::-1]
        embr_pr_wor = tparams['word_emb'][ctzr_wor.flatten()]
        embr_pr_wor = embr_pr_wor.reshape([n_timesteps_pr_wor, n_samples_pr*n_samples_qu, options['dim_word']])
        EntityPredicateEncoderInsPrr = EntityPredicateEncoder()
        projr_pr_wor = EntityPredicateEncoderInsPrr.encoder(tparams,options,prefix='encoderr_pr_wor',state_below=embr_pr_wor,mask=chzr_mask_wor)

        ctx_pr_wor = (tensor.sum((tensor.concatenate([proj_pr_wor, projr_pr_wor[::-1]], axis=proj_pr_wor.ndim-1)*chz_mask_wor[:,:,None]),axis=0)/((tensor.sum(chz_mask_wor,axis=0)+fill_chz_maks_matrix)[:,None])).reshape([n_samples_pr,n_samples_qu,options['dim_word']])

      
        ####target attention for relation score , word score, character score
 
        AttentionInsRel = AttentionLayer(tparams,prefix='att_rel',x=ctx_qu_second)
        ctx_qu_rel,ctx_qu_rel_alpha = AttentionInsRel.AttentionProc(tparams,prefix='att_rel',h=ctx_pr_rel,x=ctx_qu_second,mask = x_mask)
        AttentionInsWor = AttentionLayer(tparams,prefix='att_wor',x=ctx_qu_second)
        ctx_qu_wor,ctx_qu_wor_alpha = AttentionInsWor.AttentionProc(tparams,prefix='att_wor',h=ctx_pr_wor,x=ctx_qu_second,mask = x_mask)
        AttentionInsCha = AttentionLayer(tparams,prefix='att_cha',x=ctx_qu_second)
        ctx_qu_cha,ctx_qu_cha_alpha = AttentionInsCha.AttentionProc(tparams,prefix='att_cha',h=ctx_pr_cha,x=ctx_qu_second,mask = x_mask)

    
        ###calculate last relation score
        z_mask_untidy = tensor.max(chz_mask_cha, axis=0).reshape([n_samples_pr, n_samples_qu])
        p_rel,p_wor,p_cha,p_pr,p_max_pr = SemanticRelevanceLayerIns.CalculateCosineCross_withoutQA(options,ctx_qu_rel,ctx_qu_wor,ctx_qu_cha,ctx_pr_rel,ctx_pr_wor,ctx_pr_cha,z_mask_untidy,pattern='RWC')
        
        CriterionIns = Criterions()
        errors = CriterionIns.errors(p_max_en,p_max_pr,t)
        en_errors = CriterionIns.en_errors(p_max_en,t)
        pr_errors = CriterionIns.pr_errors(p_max_pr,t)

        return x, x_mask, y, y_mask,z_rel,z_mask_rel,z_wor,z_mask_wor,z_cha,z_mask_cha,t,errors,en_errors,pr_errors