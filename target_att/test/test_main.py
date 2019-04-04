import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import cPickle as pkl
import pdb as ipdb
import numpy
import copy
import os
import warnings
import time
from modules.build_model import MODEL
from data_analysis.prepare_date import PrepareDate
from data_analysis.data_iterator import TextIterator
from collections import OrderedDict

def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        # print kk
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams

def test(
        paramsets = '../model/model.npz',
        optionssets = '../model/model.npz.pkl',
        test_datasets = ['../datasets/simQA_test.txt',
                         '../datasets/cand_ent_test.txt',
                         '../datasets/cand_rel_test.txt'],
        dict_word = '../datasets/dict/dict.pkl',
        dict_character = '../datasets/dict/dict.pkl',
        dict_relation = '',
        predicate_num = 150,
        test_batch_size = 16,
        result_file = ''):
    print 'Loading options...',
    with open(optionssets,'rb') as f:
        model_options = pkl.load(f)
    print 'Done'
    # model_options['relation_pattern'] = 'RWC'
    print 'Loading params...',
    params = numpy.load(paramsets)

    print 'Loading data...',
    test = TextIterator(test_datasets[0],test_datasets[1], test_datasets[2],
                    dict_character,dict_word,dict_relation, predicate_num = predicate_num, batch_size=model_options['batch_size'], maxlen=model_options['maxlen'])

    print 'Done'
    tparams = init_tparams(params)
    print 'Building Model...'
    ModelIns = MODEL()

    x_v, x_mask_v, y_v, y_mask_v,z_rel_v,z_mask_rel_v,z_wor_v,chz_mask_wor_v,z_cha_v,chz_mask_cha_v,t_v,errors_v,en_errors_v,pr_errors_v= ModelIns.BuildValidTestModel(tparams,model_options)
    inputs = [x_v, x_mask_v, y_v, y_mask_v,z_rel_v,z_mask_rel_v,z_wor_v,chz_mask_wor_v,z_cha_v,chz_mask_cha_v,t_v]
    outputs = [errors_v,en_errors_v,pr_errors_v]
    # outputs_alpha = [ctx_qu_rel_alpha,ctx_qu_wor_alpha,ctx_qu_cha_alpha]
    print 'Building error function...'
    func_valid_error = theano.function(inputs,outputs,on_unused_input = 'ignore',allow_input_downcast=True)
    print 'Done'
 
    rights = []
    rights_en = []
    rights_pr = []

    for source, target,entity, predicate_relation,predicate_word,predicate_character in test:
        test_prepare_layer = PrepareDate(source,entity,predicate_character)
        x, x_mask, y, y_mask, z_relation, \
        z_mask_relation,z_word, z_mask_word,z_character, \
        z_mask_character,t= test_prepare_layer.prepare_valid_test_date_for_cross(source, entity,
                                                                     predicate_relation,predicate_word,
                                                                     predicate_character,target)
        right = func_valid_error(x, x_mask, y, y_mask, z_relation,
        z_mask_relation,z_word, z_mask_word,z_character,
        z_mask_character,t)

        rights.append(right[0])
        rights_en.append(right[1])
        rights_pr.append(right[2])
        # alpha.append(alpha_)
    right_arr = numpy.array(rights).sum()/21687.0
    right_arr_en = numpy.array(rights_en).sum()/21687.0
    right_arr_pr = numpy.array(rights_pr).sum()/21687.0
    print right_arr,right_arr_en,right_arr_pr
    return right_arr,right_arr_en,right_arr_pr
if __name__ == '__main__':
    pass
