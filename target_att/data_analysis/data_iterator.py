import numpy
import re
import cPickle as pkl
import gzip


def fopen(filename, mode='r'):
    if filename.endswith('.gz'):
        return gzip.open(filename, mode)
    return open(filename, mode)


class TextIterator:
    """Simple Bitext iterator."""
    def __init__(self, source, entity, predicate,
                 dict_character,
                 dict_word,
                 dict_relation,
                 predicate_num = 150,
                 batch_size=128,
                 maxlen=100,
                 n_words_source=-1,
                 n_words_entity=-1,
                 n_words_predicate=-1):
        self.source = fopen(source, 'r')
        self.entity = fopen(entity, 'r')
        self.predicate = fopen(predicate,'r')

        with open(dict_character, 'rb') as f1:
            self.dict_character = pkl.load(f1)
        with open(dict_word, 'rb') as f2:
            self.dict_word = pkl.load(f2)
        with open(dict_relation, 'rb') as f3:
            self.dict_relation = pkl.load(f3)

        self.batch_size = batch_size
        self.maxlen = maxlen

        self.n_words_source = n_words_source
        self.n_words_entity = n_words_entity
        self.n_words_predicate =n_words_predicate

        self.source_buffer = []
        self.target_buffer = []
        self.entity_buffer = []
        self.predicate_buffer_relation = []
        self.predicate_buffer_word = []
        self.predicate_buffer_character = []

        self.k = batch_size * 20
        self.predicate_num = predicate_num

        self.end_of_data = False

    def __iter__(self):
        return self

    def reset(self):
        self.source.seek(0)
        self.entity.seek(0)
        self.predicate.seek(0)

    def next(self):
        if self.end_of_data:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        source = []
        target = []
        entity = []
        predicate_relation = []
        predicate_word=[]
        predicate_character =[]

        # fill buffer, if it's empty
        assert len(self.source_buffer) == len(self.entity_buffer), 'Buffer size mismatch!'
        text_ = 0
        dd=0
        if len(self.source_buffer) == 0:

            for k_ in xrange(self.k):
                text_ =text_+1
                ss = self.source.readline() 
                if ss == "":
                    break
                ee = self.entity.readline()
                if ee == "":
                    break
                pp = self.predicate.readline()
                if pp == "":
                    break
                a=[]
                target1 =[]
                b=[]
                c_relation=[]
                c_word=[]
                c_character=[]
                a_f = filter(None, ss.strip('\n').split(' @#$ '))
                b_f = filter(None, ee.strip('\n').split(' @#$ '))
                try:
                    b_f.remove('')
                except ValueError:
                    pass
                if len(b_f)>500:
                    continue
                # print b_f
                ###########################################
                # c_f=filter(None,pp.strip('\n').split(' @#$ '))
                ###########################################
                c_f_ = pp.strip('\n').split(' @$@ ')
                try:
                    c_f_.remove('')
                except ValueError:
                    pass
                c_f = []
                for cur_c in c_f_:
                    c_f_2 =[]
                    c_f_f = cur_c.strip().split(' @#$ ')
                    try:
                        c_f_f.remove('')
                    except ValueError:
                        pass

                    for cc_f in c_f_f:
                        if cc_f != '':
                            c_f_2.append(cc_f)
                    c_f.append(c_f_2)
                #############################################

                for aa in a_f[0]:
                    if aa == '$' or aa == '?':
                        break
                    if aa != '':
                        a.append(aa)
                a.append('?')
                # print a

                if a_f[1] in b_f:
                    a_en = b_f.index(a_f[1])
                else:
                    a_en = -1
                if a_en != -1:
                    t_r_set = c_f[a_en]
                    if a_f[2] in t_r_set:
                        a_pr = t_r_set.index(a_f[2])
                    else:
                        print ss
                        print t_r_set
                        a_pr = -1
                else:
                    a_pr = -1
                for bb in b_f:
                    if bb != '':
                        b.append(bbb for bbb in bb)
                for cc in c_f:
                    cc_relation = []
                    cc_word = []
                    cc_character = []
                    for cc_2 in cc:
                        cc_relation_2 = []
                        cc_word_2 = []
                        cc_character_2 = []

                        if cc_2 != '':
                            # relation level
                            cc_relation_2.append(cc_2)

                            # word level
                            ccc = re.split(r'([_/])', cc_2)
                            # print 'ccc',ccc
                            for ccc_word in ccc:
                                if ccc_word !='':
                                    cc_word_2.append(ccc_word)

                            # character level
                            for c_cha in cc_2:
                                cc_character_2.append(c_cha)

                        cc_relation.append(cc_relation_2)
                        cc_word.append(cc_word_2)
                        cc_character.append(cc_character_2)

                    c_relation.append(cc_relation)
                    c_word.append(cc_word)
                    c_character.append(cc_character)

                target1.append(a_en)
                target1.append(a_pr)
                self.source_buffer.append(a)
                self.target_buffer.append(target1)
                self.entity_buffer.append(b)
                self.predicate_buffer_relation.append(c_relation)
                self.predicate_buffer_word.append(c_word)
                self.predicate_buffer_character.append(c_character)

            # sort by source buffer
            slen = numpy.array([len(t) for t in self.source_buffer])
            sidx = slen.argsort()

            _sbuf = [self.source_buffer[i] for i in sidx]
            _tbuf = [self.target_buffer[i] for i in sidx]
            _ebuf = [self.entity_buffer[i] for i in sidx]
            _pbuf_r = [self.predicate_buffer_relation[i] for i in sidx]
            _pbuf_w = [self.predicate_buffer_word[i] for i in sidx]
            _pbuf_c = [self.predicate_buffer_character[i] for i in sidx]
            self.source_buffer = _sbuf
            self.target_buffer = _tbuf
            self.entity_buffer = _ebuf
            self.predicate_buffer_relation = _pbuf_r
            self.predicate_buffer_word = _pbuf_w
            self.predicate_buffer_character = _pbuf_c

        if len(self.source_buffer) == 0 or len(self.entity_buffer) == 0 or len(self.predicate_buffer_relation) == 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        try:

            # actual work here
            k=0
            dd=0
            while True:

                # read from source file and map to word index
                try:
                    ss = self.source_buffer.pop()
                except IndexError:
                    break
                ss = [self.dict_character[w] if w in self.dict_character else 1
                      for w in ss]
                if self.n_words_source > 0:
                    ss = [w if w < self.n_words_source else 1 for w in ss]

                # read from target file and map to word index
                tt = self.target_buffer.pop()
                # for ii, ww in enumerate(tt):
                #     tt[ii] = [self.dict[w] if w in self.dict else 1
                #         for w in ww]

                # read from entity file and map to word index
                ee = self.entity_buffer.pop()
                for ii, ww in enumerate(ee):
                    ee[ii] = [self.dict_character[w] if w in self.dict_character else 1
                        for w in ww]
                    if self.n_words_entity > 0:
                        ee[ii] = [w if w < self.n_words_entity else 1 for w in ee[ii]]

                # read from predicate file and map to word index
                pp_relation = self.predicate_buffer_relation.pop()

                for ii, ww in enumerate(pp_relation):
                    for jj, www in enumerate(ww):
                        pp_relation[ii][jj] = [self.dict_relation[w] if w in self.dict_relation else 1
                            for w in www]
                    # if self.n_words_predicate > 0:
                    #     pp_relation[ii] = [w if w < self.n_words_predicate else 1 for w in pp_relation[ii]]
                pp_word = self.predicate_buffer_word.pop()
                for ii, ww in enumerate(pp_word):
                    for jj, www in enumerate(ww):
                        pp_word[ii][jj] = [self.dict_word[w] if w in self.dict_word else 1
                            for w in www]
                    # if self.n_words_predicate > 0:
                    #     pp_word[ii] = [w if w < self.n_words_predicate else 1 for w in pp_word[ii]]
                pp_character = self.predicate_buffer_character.pop()
                for ii, ww in enumerate(pp_character):
                    for jj, www in enumerate(ww):
                        pp_character[ii][jj] = [self.dict_character[w] if w in self.dict_character else 1
                            for w in www]
                    # if self.n_words_predicate > 0:
                    #     pp_character[ii] = [w if w < self.n_characters_predicate else 1 for w in pp_character[ii]]
                if len(ss) > self.maxlen and len(ee) > self.maxlen and len(pp_character) >self.maxlen:
                    continue

                source.append(ss)
                target.append(tt)
                entity.append(ee)
                predicate_relation.append(pp_relation)
                predicate_word.append(pp_word)
                predicate_character.append(pp_character)

                if len(source) >= self.batch_size or \
                        len(entity) >= self.batch_size or \
                        len(predicate_relation) >= self.batch_size:
                    break
        except IOError:
            self.end_of_data = True


        if len(source) <= 0 or len(entity) <= 0 or len(predicate_relation) <= 0:
            self.end_of_data = False
            self.reset()
            raise StopIteration

        return source, target,entity, predicate_relation,predicate_word,predicate_character
