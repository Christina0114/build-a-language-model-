
import sys
from math import log, isclose
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np  # numpy provides useful maths and vector operations
from numpy.random import random_sample
import re
import sys
from random import random
from math import log
from collections import defaultdict


#s1882930
#s1842611


# judge the character belongs to the alphabet
def is_alphabet(char):
    if (char >= 'a' and char <= 'z'
	            or
	    char >= 'A' and char <= 'Z'  ):

        return True
    else:
        return False


def preprocess_line(a):
    b = ['##']
     # set   ‘##’as start of a sentence
     # in order to get the probability of first character when we use bigram
     # and trigram model
    for i in range(len(a)):
        if a[i].isdigit():
            b.append('0')
        if a[i] == " " or a[i] == '.' or is_alphabet(a[i]):
            b.append(a[i])
    b.append('#')
     # set   ‘#’ as end of a sentence
    c = ''.join(b).lower()
    return c



# get the ngram model and compute the counts of n_grams from the training set
def ngram_counts(n, file):
    counts = defaultdict(int)
    line_count = 0
    with open(file) as f:
        for line in f:
            line = preprocess_line(line)
            line_count += 1
            if len(line) > n:
                for i in range(3 - n, len(line) - (n - 1), 1):
                    gram = line[i:i + n]
                    counts[gram] += 1  # ngram model counts
    return counts, line_count


#extract the model given by teachers
with open("/Users/chenwenjun/PycharmProjects/w2/venv/model-br.en") as f:
    tr_frombr = defaultdict(float)
    for line in f:
        trigram = line[0:3]
        # print(trigram)
        p = float(line[4:-1])
        tr_frombr[trigram] = p


# calculate the unigram probability
#the input is the counts of unigrams in the training set
def u_p(origin_counts):
    p_model = defaultdict(float)
    tokens = 0
    types = 0
    for i in origin_counts:
        tokens += origin_counts[i]
    for i in origin_counts:
        p_model[i] = origin_counts[i] / tokens
    types = len(origin_counts)
    return p_model


	#uc is an dictionary contains the the counts of unigrams and probabilities
	#origin_b_c is the counts of bigrams in the training set
	#return the the whole dictionary containing all bigrams

def b_model_cor(uc, origin_b_c):
    bmc = defaultdict(int)
    for i in uc:
        for j in uc:
            temp = []
            temp.append(i)
            temp.append(j)
            word = ''.join(map(str, temp))
            if (word in origin_b_c.keys()):
                bmc[word] = origin_b_c[word]
            else:
                bmc[word] = 0
    del bmc['##']
    return bmc


	# get  the counts of trigrams
	# tc is the original counts of the trigram in the training set
def t_model_cor(uc, tc):
    tmc = defaultdict(int)
    for i in uc:
        for j in uc:
            for m in uc:
                temp = []
                temp.append(i)
                temp.append(j)
                temp.append(m)
                word = ''.join(map(str, temp))
                if (word in tc.keys()):
                    tmc[word] = tc[word]
                else:
                    tmc[word] = 0
    for i in uc:
        for j in uc:
            for m in uc:
                if (i == '#' and m == '#'):
                    del tmc[i + j + m]
    for i in uc:
        for j in uc:
            for m in uc:
                if (i != '#'):
                    if (j == '#'):
                        del tmc[i + j + m]

    return tmc


	#  bcp is a whole counts for bigram dictionary
def b_gram(bcp, uc):
    p_model = defaultdict(float)
    tokens = 0
    types = len(bcp)
    for i in bcp:
        tokens += bcp[i]
    for i in bcp:
        p_model[i] = bcp[i] / uc[i[0]]
    return p_model


	#set up the trigram model contain the zero probabilities
	#tcp is a dictionary containing the probability of the trigrams
	#bcp is a dictionary containing the probability of the bigrams

def t_gram(tcp, bcp):
    p_model = defaultdict(float)
    tokens = 0
    types = len(tcp)
    line_number = 0
    for i in tcp:
        tokens += tcp[i]
    for i in tcp:
        if (i[:2] == '##'):
            line_number = line_number + tcor_c[i]  # calculate the number of lines
    for i in tcp:
        if (bcp[i[:2]] != 0):
            p_model[i] = tcp[i] / bcp[i[:2]]
        else:
            if (i[:2] == '##'):
                p_model[i] = tcp[i] / line_number
            else:
                p_model[i] = 0
    return p_model

	# using the a1,a2,a3 to set up the interpolation
def interplotaion(u, b, t, a1, a2, a3):
    p_model = defaultdict(float)
    for i in t:
        p_model[i] = a1 * t[i] + a2 * b[i[1:]] + a3 * u[i[2:]]
    return p_model



def perplexity(file, n, model):
    counts = defaultdict(int)
  # set up a dictionary which contains the probabilities and the trigrams
    with open(file) as f:
        for line in f:
            line = preprocess_line(line)
            if len(line) > n:
                for i in range(3 - n, len(line) - (n - 1), 1):
                    gram = line[i:i + n]
                    if (model[gram] > 0):
                        counts[gram] = -log(model[gram], 2)
                     # compute the log and add them together
                       # print(gram)
                    else:
                        counts[gram] = 0  # ngram model counts
    return  2**(sum(counts.values()) / len(counts))


def generate_from_LM(distribution,N):
    y=['r','e','s','u','m','p','t','i','o','n',' ',
        'f','h','d','c','l','a','j','y','0','b','w','k','g','v','.','q','x','z']
    generate=[]
    w='##'
    for j in range(N):
        outcomes=[]
        probs=[]
        for i in y :
            a=[w[-2:]]
            a.append(i)
            a=''.join(a)
            outcomes.append(a)
            probs.append(distribution.get(a))
        probs = np.array(probs)
        summ=np.sum(probs)
        probs=probs/summ
        index = np.random.choice(outcomes, p = probs.ravel())
        generate.append(index)
        w=w+generate[-1][-1:]
    return print(w)





# -----------------------------------------------------------
	# generate the basic counts dictionary
u_counts, line_count = ngram_counts(1, "/Users/chenwenjun/PycharmProjects/w2/venv/training.en")
b_counts, line_count = ngram_counts(2, "/Users/chenwenjun/PycharmProjects/w2/venv/training.en")
t_counts, line_count = ngram_counts(3, "/Users/chenwenjun/PycharmProjects/w2/venv/training.en")

	#generate the unigram model
ucor_c = defaultdict(int)
ucor_c = u_counts
# print((ucor_c))

# generate unigram model
ucor_p = defaultdict(float)
ucor_p = u_p(ucor_c)


	# generate bigram model
bcor_c = defaultdict(int)
bcor_c = b_model_cor(u_counts, b_counts)

bcor_p = defaultdict(float)
bcor_p = b_gram(bcor_c, u_counts)

tcor_c = defaultdict(int)
tcor_c = t_model_cor(u_counts, t_counts)

tcor_p = defaultdict(float)
tcor_p = t_gram(tcor_c, bcor_c)

# generate interpolation
f1 = defaultdict(float)
f1 = interplotaion(ucor_p, bcor_p, tcor_p, 1/3, 1/3, 1/3)
#print(f1)

ng_d=defaultdict(float)
for i in f1:
    if i[:2]=='ng':
       ng_d[i]=f1[i]


# ------------------------
#calculate the some features of brain.en document"
print(max(tr_frombr.values()))
print(min(tr_frombr.values()))
#print(tr_frombr)
nnn = 0
for i in tr_frombr:
    if tr_frombr[i]== 0.03333:
        nnn=nnn+1
#calculate the number of 1/3


#-----------------------------
#generate the sequence
generate_from_LM(f1,300)
generate_from_LM(tr_frombr,300)

#-----------------------------
print(perplexity('/Users/chenwenjun/PycharmProjects/w2/venv/test', 3, f1))
print(perplexity('/Users/chenwenjun/PycharmProjects/w2/venv/training.es', 3, f1))
print(perplexity('/Users/chenwenjun/PycharmProjects/w2/venv/training.de', 3, f1))

#--------------


	#min is the lowest perplexity
	#min_i,min_j,min_m represent the parameter  λ1,λ_2,λ_3
min=10000
min_i=min_j=min_m=0
for i in np.arange(0.1,0.9,0.01):
	for j in np.arange(0.1, 0.9, 0.01):
	    if(i+j<1):
	        m=1-i-j
	        f1 = interplotaion(ucor_p, bcor_p, tcor_p, i, j, m)
	        per=perplexity('/Users/chenwenjun/PycharmProjects/w2/venv/validation_set', 3, f1)
	        if(per<min):
	            min=per
	            min_i=i
	            min_j=j
	            min_m = m
#-------------------------------#
#develop our model
f2=interplotaion(ucor_p, bcor_p, tcor_p, 0.72, 0.15, 0.13)
print(perplexity('/Users/chenwenjun/PycharmProjects/w2/venv/test', 3, f2))
print(perplexity('/Users/chenwenjun/PycharmProjects/w2/venv/training.es', 3, f2))
print(perplexity('/Users/chenwenjun/PycharmProjects/w2/venv/training.de', 3, f2))
