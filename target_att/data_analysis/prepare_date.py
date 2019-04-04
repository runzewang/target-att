import numpy
class PrepareDate(object):
    def __init__(self,seqs_x,seqs_y,seqs_z):
        self.lengths_x=[]
        self.lengths_y=[]
        self.lengths_z=[]
        self.n_samples = len(seqs_x)
        self.en_num = len(seqs_y[0])
        self.pr_num = len(seqs_z[0])
        self.one_pr_num = self.pr_num/self.en_num
        self.maxlen_x = 0
        self.maxlen_e = 0
        self.maxlen_p = 0

    def prepare_valid_test_date_for_cross(self, seqs_x, seqs_y, seqs_z_relation, seqs_z_word, seqs_z_character, seqs_t):
        for s in seqs_x:
            self.lengths_x.append(len(s))

        maxlen_ee = []
        maxnum_ee = []


        maxlen_pp_relation = []
        maxlen_pp_word = []
        maxlen_pp_character = []

        maxnum_pp = []


        lengths_z_relation = []
        lengths_z_word = []
        lengths_z_character =[]


        for ii in range(len(seqs_y)):
            ee = [len(s) for s in seqs_y[ii]]
            self.lengths_y.append(ee)
            maxlen_ee.append(numpy.max(ee))
            maxnum_ee.append(len(ee))
        maxnum_e = numpy.max(maxnum_ee)

        for ii in range(len(seqs_z_relation)):
            lengths_z_relation_2 = []
            for iii in range(len(seqs_z_relation[ii])):
                pp_relation = [len(ss) for ss in seqs_z_relation[ii][iii]]
                lengths_z_relation_2.append(pp_relation)
                maxlen_pp_relation.append(numpy.max(pp_relation))
                maxnum_pp.append(len(pp_relation))
            lengths_z_relation.append(lengths_z_relation_2)
            
        maxlen_p_relation = numpy.max(maxlen_pp_relation)
        maxnum_p = numpy.max(maxnum_pp)

        for ii in range(len(seqs_z_word)):
            lengths_z_word_2 = []
            for iii in range(len(seqs_z_word[ii])):
                pp_word = [len(ss) for ss in seqs_z_word[ii][iii]]
                lengths_z_word_2.append(pp_word)
                maxlen_pp_word.append(numpy.max(pp_word))
            lengths_z_word.append(lengths_z_word_2)
        maxlen_p_word = numpy.max(maxlen_pp_word)

        for ii in range(len(seqs_z_character)):
            lengths_z_character_2 = []
            for iii in range(len(seqs_z_character[ii])):
                pp_character = [len(ss) for ss in seqs_z_character[ii][iii]]
                lengths_z_character_2.append(pp_character)
                maxlen_pp_character.append(numpy.max(pp_character))
            lengths_z_character.append(lengths_z_character_2)
        maxlen_p_character = numpy.max(maxlen_pp_character)

        self.maxlen_x = numpy.max(self.lengths_x)
        self.maxlen_e = numpy.max(maxlen_ee)

        x =numpy.zeros((self.maxlen_x, self.n_samples)).astype('int64')
        x_mask = numpy.zeros((self.maxlen_x, self.n_samples)).astype('float32')

        y = numpy.zeros((self.maxlen_e, maxnum_e*self.n_samples)).astype('int64')
        y_mask = numpy.zeros((self.maxlen_e, maxnum_e*self.n_samples)).astype('float32')

        z_relation = numpy.zeros((maxlen_p_relation, maxnum_p,maxnum_e,self.n_samples)).astype('int64')
        z_mask_relation = numpy.zeros((maxlen_p_relation, maxnum_p,maxnum_e,self.n_samples)).astype('float32')
        z_word = numpy.zeros((maxlen_p_word, maxnum_p,maxnum_e,self.n_samples)).astype('int64')
        z_mask_word = numpy.zeros((maxlen_p_word, maxnum_p,maxnum_e,self.n_samples)).astype('float32')
        z_character = numpy.zeros((maxlen_p_character,maxnum_p,maxnum_e,self.n_samples)).astype('int64')
        z_mask_character = numpy.zeros((maxlen_p_character, maxnum_p,maxnum_e,self.n_samples)).astype('float32')
        t = numpy.zeros((2, self.n_samples)).astype('int64')

        for idx, s_x in enumerate(seqs_x):
            t[0,idx] = seqs_t[idx][0]
            t[1,idx] = seqs_t[idx][1]
            x[:self.lengths_x[idx], idx] = s_x
            x_mask[:self.lengths_x[idx]+1, idx] = 1.
            for idx_y, s_y in enumerate(seqs_y[idx]):
                y[:self.lengths_y[idx][idx_y], idx+idx_y*self.n_samples] = s_y
                y_mask[:self.lengths_y[idx][idx_y], idx+idx_y*self.n_samples] = 1.
                for idx_z, s_z in enumerate(seqs_z_relation[idx][idx_y]):
                    z_relation[:lengths_z_relation[idx][idx_y][idx_z], idx_z, idx_y, idx] = s_z
                    z_mask_relation[:lengths_z_relation[idx][idx_y][idx_z], idx_z, idx_y, idx] = 1.
                for idx_z, s_z in enumerate(seqs_z_word[idx][idx_y]):
                    z_word[:lengths_z_word[idx][idx_y][idx_z], idx_z, idx_y, idx] = s_z
                    z_mask_word[:lengths_z_word[idx][idx_y][idx_z], idx_z, idx_y, idx] = 1.
                for idx_z, s_z in enumerate(seqs_z_character[idx][idx_y]):
                    z_character[:lengths_z_character[idx][idx_y][idx_z], idx_z, idx_y, idx] = s_z
                    z_mask_character[:lengths_z_character[idx][idx_y][idx_z], idx_z, idx_y, idx] = 1.
        return x, x_mask, y, y_mask, z_relation, z_mask_relation,z_word, z_mask_word,z_character, z_mask_character,t