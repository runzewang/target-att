from modules.encoder_layers.init_encoder_layers import InitLayersParamsEn
from modules.decoder_layers.init_decoder_layers import InitLayersParamsDe
from collections import OrderedDict
import theano
class InitParams:
    def __init__(self):
        self.params = OrderedDict()
        pass
    def initparams(self,options):
        InitLayersParamsEnIns = InitLayersParamsEn()
        InitLayersParamsDeIns = InitLayersParamsDe()

        self.params = InitLayersParamsEnIns.init_question_encoder_lstm(options,self.params,prefix='encoder_qu_first',nin=options['dim_word'],dim=options['dim'])
        self.params = InitLayersParamsEnIns.init_question_encoder_lstm(options,self.params,prefix='encoder_qur_first',nin=options['dim_word'],dim=options['dim'])
        self.params = InitLayersParamsEnIns.init_question_encoder_lstm(options,self.params,prefix='encoder_qu_second',nin=options['dim_word'],dim=options['dim'])
        self.params = InitLayersParamsEnIns.init_question_encoder_lstm(options,self.params,prefix='encoder_qur_second',nin=options['dim_word'],dim=options['dim'])

        self.params = InitLayersParamsEnIns.init_entity_encoder_lstm(options,self.params,prefix='encoder_en',nin=options['dim_word'],dim=options['dim'])
        self.params = InitLayersParamsEnIns.init_entity_encoder_lstm(options,self.params,prefix='encoderr_en',nin=options['dim_word'],dim=options['dim'])

        self.params = InitLayersParamsEnIns.init_embed(options,self.params,prefix='character',num=options['character_dict_num'],nin=options['dim_word'])
        self.params = InitLayersParamsEnIns.init_predicate_encoder_lstm(options,self.params,prefix='encoder_pr_cha',nin=options['dim_word'],dim=options['dim'])
        self.params = InitLayersParamsEnIns.init_predicate_encoder_lstm(options,self.params,prefix='encoderr_pr_cha',nin=options['dim_word'],dim=options['dim'])
    
        self.params = InitLayersParamsEnIns.init_embed(options,self.params,prefix='relation',num=options['relation_dict_num'],nin=options['dim_word'])
        self.params = InitLayersParamsEnIns.init_predicate_encoder_lstm(options,self.params,prefix='encoder_pr_rel',nin=options['dim_word'],dim=options['dim'])
        self.params = InitLayersParamsEnIns.init_predicate_encoder_lstm(options,self.params,prefix='encoderr_pr_rel',nin=options['dim_word'],dim=options['dim'])
    
        self.params = InitLayersParamsEnIns.init_embed(options,self.params,prefix='word',num=options['word_dict_num'],nin=options['dim_word'])
        self.params = InitLayersParamsEnIns.init_predicate_encoder_lstm(options,self.params,prefix='encoder_pr_wor',nin=options['dim_word'],dim=options['dim'])
        self.params = InitLayersParamsEnIns.init_predicate_encoder_lstm(options,self.params,prefix='encoderr_pr_wor',nin=options['dim_word'],dim=options['dim'])

        dimctx = 4*options['dim']

        self.params = InitLayersParamsDeIns.init_attention(options,self.params,prefix='decoder_en',nin=options['dim'],dim=options['dim_word'],dimctx=dimctx)
        self.params = InitLayersParamsDeIns.init_decoder(options,self.params,prefix='decoder',nin=options['dim_word'],dim=options['dim_word'],dimctx=2*dimctx)

        self.params = InitLayersParamsDeIns.init_attention(options,self.params,prefix='att_rel',nin=options['dim'],dim=2*options['dim'],dimctx=2*options['dim'])
        self.params = InitLayersParamsDeIns.init_attention(options,self.params,prefix='att_wor',nin=options['dim'],dim=2*options['dim'],dimctx=2*options['dim'])
        self.params = InitLayersParamsDeIns.init_attention(options,self.params,prefix='att_cha',nin=options['dim'],dim=2*options['dim'],dimctx=2*options['dim'])
           
        return 0
    def inittparams(self,options):
        self.initparams(options)
        tparams = OrderedDict()
        for kk, pp in self.params.iteritems():
            # print kk
            tparams[kk] = theano.shared(self.params[kk], name=kk)
        return tparams