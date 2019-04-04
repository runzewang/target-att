import numpy
import os

from test_main import test

def main():
    test_datasets=[
        'datasets/simpleqa/test_5m.question',
        'datasets/simpleqa/test_5m.subject',
        'datasets/simpleqa/test_5m.relation']
    testerr = test(
        paramsets = 'datasets/result_model/best_model_5m.npz',
        optionssets = 'datasets/result_model/best_model_5m.npz.pkl',
        test_datasets = test_datasets,
        dict_word = 'datasets/dict/dict_word.pkl',
        dict_character = 'datasets/dict/dict_character.pkl',
        dict_relation = 'datasets/dict/dict_relation.pkl',
        test_batch_size=16
    )
    print 'End Test'
    return 0

if __name__ == '__main__':
    main()