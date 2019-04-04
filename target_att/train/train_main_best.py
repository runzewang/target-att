import theano
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import cPickle as pkl
import pdb as ipdb
import numpy
import copy
import os
import warnings
import sys
import time
from collections import OrderedDict
from theano.compile.nanguardmode import NanGuardMode
from data_analysis.prepare_date import PrepareDate
from modules.build_model import MODEL
from data_analysis.data_iterator import TextIterator
from modules.init import InitParams
# get the list of parameters: Note that tparams must be OrderedDict
def itemlist(tparams):
    return [vv for kk, vv in tparams.iteritems()]
def unzip(zipped):
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params
# push parameters to Theano shared variables
def zipp(params, tparams):
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)
def adadelta(lr, tparams, grads, inp, cost):
    zipped_grads = [theano.shared(p.get_value() * numpy.float32(0.),
                                  name='%s_grad' % k)
                    for k, p in tparams.iteritems()]
    running_up2 = [theano.shared(p.get_value() * numpy.float32(0.),
                                 name='%s_rup2' % k)
                   for k, p in tparams.iteritems()]
    running_grads2 = [theano.shared(p.get_value() * numpy.float32(0.),
                                    name='%s_rgrad2' % k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2))
             for rg2, g in zip(running_grads2, grads)]

    f_grad_shared = theano.function(inp, cost, updates=zgup+rg2up,
                                    on_unused_input='ignore')
    # f_grad_shared = theano.function(inp, cost, updates=zgup+rg2up,
    #                                 on_unused_input='ignore',mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True))

    updir = [-tensor.sqrt(ru2 + 1e-6) / tensor.sqrt(rg2 + 1e-6) * zg
             for zg, ru2, rg2 in zip(zipped_grads, running_up2,
                                     running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2))
             for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(itemlist(tparams), updir)]

    f_update = theano.function([lr], [], updates=ru2up+param_up,
                               on_unused_input='ignore')

    return f_grad_shared, f_update
def detect_nan(i, node, fn):
    for output in fn.outputs:
        if (not isinstance(output[0], numpy.random.RandomState) and
            numpy.isnan(output[0]).any()):
            print '*** NaN detected ***'
            theano.printing.debugprint(node)
            print 'Inputs : %s' % [input[0] for input in fn.inputs]
            print 'Outputs: %s' % [output[0] for output in fn.outputs]
            break

def trainer(
        r = 5,
        dim_word=1000,
        dim=1000,
        trainpath=[
            '../datasets/simQA_test.txt',
            '../datasets/cand_ent_test.txt',
            '../datasets/cand_rel_test.txt'],
        validpath=[
            '../datasets/simQA_test.txt',
            '../datasets/cand_ent_test.txt',
            '../datasets/cand_rel_test.txt'],
        dict_character = '../datasets/dict/dict.pkl',
        dict_relation = '../datasets/dict/dict.pkl',
        dict_word = '../datasets/dict/dict.pkl',
        relation_pattern='RWC',
        batch_size =16,
        valid_batch_size = 16,
        maxlen=200,
        learning_rate =0.001,
        max_epochs = 10,
        dispFreq = 100,
        saveFreq = 1000,
        validFreq = 1000,
        saveto = 'model.npz',
        overwrite = True,
        patience = 10,
        predicate_num = 150,
        lstm_end = 'average',
        lstm_layers = 2,
        word = False,
        word_dict_num = 5000,
        relation_dict_num = 8000,
        character_dict_num = 200,
        cross = True,
        one_layer = False,
        en_decode_type='ff',
        qu_split = False,
        structure_number = 3,
        en_pooling_type='average',# only for pooling question when entity decoding
        relation_attention='target_attention'
        ):
    # theano.config.warn_float64 = "raise"
    model_options = locals().copy()
    train = TextIterator(trainpath[0],trainpath[1],trainpath[2],
                        dict_character,dict_word,dict_relation,predicate_num = predicate_num,batch_size=model_options['batch_size'],maxlen=model_options['maxlen'])

    valid = TextIterator(validpath[0],validpath[1],validpath[2],
                    dict_character,dict_word,dict_relation,predicate_num = predicate_num,batch_size=model_options['batch_size'],maxlen=model_options['maxlen'])

    InitParamsIns = InitParams()
    tparams = InitParamsIns.inittparams(model_options)
    ModelIns = MODEL()
    print 'Build Train and Valid Model...',
    x, x_mask, y, y_mask,z_rel,z_mask_rel,z_wor,chz_mask_wor,z_cha,chz_mask_cha,t,cost,errors = ModelIns.BuildTrainModel(tparams,model_options)
    x_v, x_mask_v, y_v, y_mask_v,z_rel_v,z_mask_rel_v,z_wor_v,chz_mask_wor_v,z_cha_v,chz_mask_cha_v,t_v,errors_v,en_errors_v,pr_errors_v= ModelIns.BuildValidTestModel(tparams,model_options)
    print 'Done'
    inputs_v = [x_v, x_mask_v, y_v, y_mask_v,z_rel_v,z_mask_rel_v,z_wor_v,chz_mask_wor_v,z_cha_v,chz_mask_cha_v,t_v]
    inputs = [x, x_mask, y, y_mask,z_rel,z_mask_rel,z_wor,chz_mask_wor,z_cha,chz_mask_cha,t]

    # alpha=[pr_alpha]
    outputs = [cost,errors]
    optputs_v =[errors_v,en_errors_v,pr_errors_v]
    func_ctx = theano.function(inputs,outputs,on_unused_input = 'ignore',allow_input_downcast=True,mode=NanGuardMode(nan_is_error=True, inf_is_error=True, big_is_error=True))
    func_valid_error = theano.function(inputs_v,optputs_v,on_unused_input = 'ignore',allow_input_downcast=True)

    # func_p = theano.function(inputs,p,on_unused_input = 'ignore',allow_input_downcast=True)
    # func_alpha = theano.function(inputs,pr_alpha,on_unused_input = 'ignore',allow_input_downcast=True)
    print 'Building grad...',
    grads = tensor.grad(cost, wrt=itemlist(tparams))
    print 'Done'

    print 'Building optimizers...',
    lr = tensor.scalar(name='lr')
    f_grad_shared, f_update = adadelta(lr, tparams, grads, inputs, cost)
    print 'Done'

    uidx =0
    best_p = None
    estop = False
    bad_counter = 0
    history_right = []
    for epoch_idx in xrange(max_epochs):
        n_samples = 0
        for source, target,entity, predicate_relation,predicate_word,predicate_character in train:
            n_samples += len(source)
            uidx += 1

            prepare_layer = PrepareDate(source,entity,predicate_character)
            x, x_mask, y, y_mask, z_relation, \
            z_mask_relation,z_word, z_mask_word,z_character, \
            z_mask_character,t = prepare_layer.prepare_valid_test_date_for_cross(source, entity,
                                                                                predicate_relation,predicate_word,
                                                                                predicate_character,target)
            if source is None:
                print 'Minibatch with zero sample'
                uidx -=1
                continue
            ud_start = time.time()

            cost = f_grad_shared(x, x_mask, y, y_mask, z_relation,
            z_mask_relation,z_word, z_mask_word,z_character,
            z_mask_character,t)
            # ctx_qu_rel,ctx_qu_wor,ctx_qu_cha,ctx_pr_rel,ctx_pr_wor,ctx_pr_cha=func_p(x, x_mask, y, y_mask, z_relation,
            # z_mask_relation,z_word, z_mask_word,z_character,
            # z_mask_character,t)
            f_update(learning_rate)
            ud = time.time()-ud_start
            if numpy.isnan(cost) or numpy.isinf(cost):
                print 'NaN detected'
                break
            if numpy.mod(uidx, dispFreq) == 0:
                print 'Epoch ', epoch_idx, 'Update ', uidx, 'Cost ', cost, 'UD ', ud,'learning_rate',learning_rate

            if numpy.mod(uidx, saveFreq) == 0:
                print 'Saving the best model...',
                if best_p is not None:
                    params = best_p
                else:
                    params = unzip(tparams)
                numpy.savez(saveto, history_errs=history_right, uidx=uidx, **params)
                pkl.dump(model_options, open('%s.pkl' % saveto, 'wb'))
                print 'Done'
                # save with uidx
                if not overwrite:
                    print 'Saving the model at iteration {0}...'.format(uidx),
                    saveto_uidx = '{0}.iter{1}.npz'.format(
                        os.path.splitext(saveto)[0], uidx)
                    numpy.savez(saveto_uidx, history_errs=history_right,
                                uidx=uidx, **unzip(tparams))
                    print 'Done'
            # validdata model on validation set and early stop if necessary
            if numpy.mod(uidx, validFreq) == 0:
                rights = []
                for source, target,entity, predicate_relation,predicate_word,predicate_character in valid:
                    valid_prepare_layer = PrepareDate(source,entity,predicate_character)
                    x, x_mask, y, y_mask, z_relation, \
                    z_mask_relation,z_word, z_mask_word,z_character, \
                    z_mask_character,t= valid_prepare_layer.prepare_valid_test_date_for_cross(source, entity,
                                                                                predicate_relation,predicate_word,
                                                                                predicate_character,target)

                    right = func_valid_error(x, x_mask, y, y_mask, z_relation,
                    z_mask_relation,z_word, z_mask_word,z_character,
                    z_mask_character,t)

                    rights.append(right[0])

                right_arr = numpy.array(rights)
                valid_right = right_arr.mean()/valid_batch_size
                history_right.append(valid_right)

                if uidx == 0 or valid_right >= numpy.array(history_right).max():
                    best_p = unzip(tparams)
                    bad_counter = 0
                if len(history_right) > patience and valid_right <= numpy.array(history_right)[:-patience].max():
                    bad_counter += 1
                    if bad_counter > patience:
                        print 'Early Stop!'
                        estop = True
                        break
                # if numpy.isnan(valid_err):
                #     ipdb.set_trace()
                print 'Valid ', valid_right
        print 'seen %d samples' % n_samples
        if estop:
            break
        print 'Saving the model at epoch {0}...'.format(epoch_idx),
        saveto_uidx = '{0}.epoch{1}.npz'.format(
            os.path.splitext(saveto)[0], epoch_idx)
        numpy.savez(saveto_uidx, history_errs=history_right,
                    uidx=uidx, **unzip(tparams))
        print 'Done'
    if best_p is not None:
        zipp(best_p, tparams)

    rights = []
    for source, target,entity, predicate_relation,predicate_word,predicate_character in valid:
        valid_prepare_layer = PrepareDate(source,entity,predicate_character)
        x, x_mask, y, y_mask, z_relation, \
        z_mask_relation,z_word, z_mask_word,z_character, \
        z_mask_character,t= valid_prepare_layer.prepare_valid_test_date_for_cross(source, entity,
                                                                    predicate_relation,predicate_word,
                                                                    predicate_character,target)
        right = func_valid_error(x, x_mask, y, y_mask, z_relation,
        z_mask_relation,z_word, z_mask_word,z_character,
        z_mask_character,t)
        rights.append(right[0])

    right_arr = numpy.array(rights)
    valid_right = right_arr.mean()/valid_batch_size

    print 'Valid ', valid_right
    # train_err =numpy.array(p_train).mean()/batch_size
    params = copy.copy(best_p)
    numpy.savez(saveto, zipped_params=best_p,
                history_errs=history_right,
                uidx=uidx,
                **params)
    return valid_right
if __name__ == '__main__':
    pass