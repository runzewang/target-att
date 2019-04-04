import numpy
import os
import socket
from train_main_best import trainer

def get_host_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip
def main():
    print 'Host Name:', get_host_ip(), 'ID:', os.getpid()
    trainpath=[
            'datasets/simpleqa/train_5m.question',
            'datasets/simpleqa/train_5m.subject',
            'datasets/simpleqa/train_5m.relation']
    validpath=[
            'datasets/simpleqa/valid_5m.question',
            'datasets/simpleqa/valid_5m.subject',
            'datasets/simpleqa/valid_5m.relation']
    f = open(trainpath[0],'r')
    ques_num = len(f.readlines())
    validFreq = int(ques_num/16) +1
    print 'ques_num', ques_num, 'validFreq', validFreq
    trainerr = trainer(
         r = 5,
         dim_word=200,
         dim=100,
         trainpath=trainpath,
         validpath=validpath,
         dict_character = 'datasets/dict/dict_character.pkl',
         dict_relation = 'datasets/dict/dict_relation.pkl',
         dict_word = 'datasets/dict/dict_word.pkl',
         batch_size =16,
         valid_batch_size=16,
         maxlen=200,
         learning_rate =0.001,
         max_epochs = 50,
         dispFreq = 100,
         saveFreq = validFreq,
         validFreq = validFreq,
         saveto='datasets/result_model/target_attention.npz',
         overwrite = False,
         patience = 3,
         predicate_num =150,
         lstm_end = 'average',
         lstm_layers= 1,
         word=False,
         word_dict_num=5000,
         relation_dict_num=8000,
         character_dict_num=200,
         cross = True,
         one_layer= False,
         structure_number = 3,
         en_pooling_type='average')
    return trainerr
if __name__ == '__main__':
    main()
