# set 3 sentences for summarization
# but it has some error when CROSS OVER step
# it has to be fixed.

from gensim.models import Word2Vec
import numpy as np
import scipy.spatial.distance as Dist
from math import log

embedding_model = Word2Vec.load('./data/word2vec/word2vec_model')
#print(embedding_model.most_similar(positive=["미국"], topn=10))

article_num =10
word_vec_size = 50
summary_length = 6

def get_tr(summary, title):
    # summary : list (element : string)
    # title : string

    #get title vector
    title_vec = np.zeros(word_vec_size)
    split_title = title.split()
    for word in split_title:
        word_vec = embedding_model.wv[word]
        title_vec += word_vec

    total_sim = 0
    tr_list = []
    for sentence in summary:
        sen_vec = np.zeros(word_vec_size)
        for word in sentence:
            try:
                word_vec = embedding_model.wv[word]
                sen_vec += word_vec
            except KeyError:
                cnt=0
        sim = Dist.cosine(title_vec, sen_vec)
        total_sim += sim
    TR = total_sim/ summary_length

    return TR
    """
    for sentence in article:
        for word in article:
    """

def get_sim_matrix(article):
    N = len(article)
    sim_mat = np.zeros([N,N])

    sen_vec_list = []
    for sentence in article:
        sen_vec = np.zeros(word_vec_size)
        for word in sentence:
            try:
                word_vec = embedding_model.wv[word]
                sen_vec += word_vec
            except KeyError:
                cnt = 0

        sen_vec_list.append(sen_vec)

    for i in range(N):
        for j in range(N):
            vec_i = sen_vec_list[i]
            vec_j = sen_vec_list[j]
            sim = Dist.cosine(vec_i, vec_j)
            sim_mat[i][j] = sim

    return sim_mat

def get_cf(summary_index, sim_mat):
    C = 0
    for i in summary_index:
        for j in summary_index:
            C += sim_mat[i][j]
    N = summary_length*(summary_length-1)/2
    C = C/N
    M = sim_mat.max()

    CF = log(C*9 + 1) / log(M*9 + 1)

    return CF

def get_fitness(artcle, summary_index, title):
    #article : list
    #summary_index : list
    #title : string

    #ratio of the metric
    alpha = 0.5
    beta = 0.5

    summary = []
    for i in summary_index:
        summary.append(article[i])

    tr = get_tr(summary, title)
    sim_mat = get_sim_matrix(article)
    cf = get_cf(summary_index, sim_mat)

    fitness = (alpha*tr + beta*cf) / (alpha+beta)
    return fitness

input_article = []
input_title = []
with open('./data/morpheme/morphemed_input.txt', 'r') as f:
    while True:
        line = f.readline()
        if not line: break
        line = line[:-1].split('\t')
        input_article.append(line)

with open('./data/morpheme/morphemed_title.txt', 'r') as f:
    while True:
        line = f.readline()
        if not line: break
        input_title.append(line[:-1])

article = input_article[article_num]
title = input_title[article_num]
#title ='노먼 교수팀 케임브리지대 지구과학부 영국'
print(type(title))

#initialize gene
gene_1 = np.zeros(len(article))
gene_2 = np.zeros(len(article))
gene_3 = np.zeros(len(article))
gene_4 = np.zeros(len(article))

index = np.random.choice(len(article), summary_length, replace=False)
gene_1[index] = 1
index = np.random.choice(len(article), summary_length, replace=False)
gene_2[index] = 1
index = np.random.choice(len(article), summary_length, replace=False)
gene_3[index] = 1
index = np.random.choice(len(article), summary_length, replace=False)
gene_4[index] = 1

print(gene_1)

genes = [gene_1, gene_2, gene_3, gene_4]

#start spreading
for itr in range(100):
    genes = [gene_1, gene_2, gene_3, gene_4]

    #fitness
    fit1 = get_fitness(article, list(np.where(gene_1 == 1)[0]), title)
    fit2 = get_fitness(article, list(np.where(gene_2 == 1)[0]), title)
    fit3 = get_fitness(article, list(np.where(gene_3 == 1)[0]), title)
    fit4 = get_fitness(article, list(np.where(gene_4 == 1)[0]), title)

    fit_list = [fit1, fit2, fit3, fit4]
    sort_list = [fit1, fit2, fit3, fit4]
    sort_list.reverse()

    print(str(itr) + ' fit1 : ' + str(fit1))
    print(str(itr) + ' fit2 : ' + str(fit2))

    best1 = fit_list.index(sort_list[0])
    best2 = fit_list.index(sort_list[1])

    best1 = genes[best1]
    best2 = genes[best2]

    best1 = list(np.where(best1==1)[0])
    best2 = list(np.where(best2==1)[0])

    tmp1 = []
    tmp2 =[]
    tmp3 = []
    tmp4 = []
    print(str(itr) + ' : cut choice')
    cut = np.random.choice(summary_length - 1, 2, replace=False)
    if cut[0] == 0: cut[0] = 1
    if cut[1] == 0: cut[1] = 1

    if best1[:cut[0]][-1] == best2[cut[0]]:
        tmp1 = list(set(best1[cut[0]:]) - set(best2[cut[0]:]))
        if tmp1: tmp1=[tmp1[0]]
        else: tmp1 = list(set(best2[cut[0]:]) - set(best1[cut[0]:]))
        if tmp1: tmp1 = [list( set(list(range(len(article)))) - (set(best2+best1)))[0]]

    if best1[:cut[1]][-1] == best2[cut[1]]:
        tmp2 = list(set(best1[cut[1]:]) - set(best2[cut[1]:]))
        if tmp2: tmp2=[tmp2[0]]
        else: tmp2 = list(set(best2[cut[1]:]) - set(best1[cut[1]:]))
        if tmp2: tmp2 = [list( set(list(range(len(article)))) - (set(best2 + best1)))[0]]

    if best2[:cut[0]][-1] == best1[cut[0]]:
        tmp3 = list(set(best2[cut[0]:]) - set(best1[cut[0]:]))
        if tmp3:
            tmp3 = [tmp3[0]]
        else:
            tmp3 = list(set(best1[cut[0]:]) - set(best2[cut[0]:]))
        if tmp3: tmp3 = [list(set(list(range(len(article)))) - (set(best2+best1)))[0]]

    if best2[:cut[1]][-1] == best1[cut[1]]:
        tmp4 = list(set(best2[cut[1]:]) - set(best1[cut[1]:]))
        if tmp4:
            tmp4 = [tmp4[0]]
        else:
            tmp4 = list(set(best1[cut[1]:]) - set(best2[cut[1]:]))
        if tmp4: tmp4 = [list(set(list(range(len(article)))) - (set(best2+best1)))[0]]


    gene1 = np.array(best1[:cut[0]]+best2[cut[0]:] +tmp1)
    gene2 = np.array(best1[:cut[1]]+best2[cut[1]:] +tmp2)

    gene3 = np.array(best2[:cut[0]]+best1[cut[0]:] +tmp3)
    gene4 = np.array(best2[:cut[1]]+best1[cut[1]:] +tmp4)

    gene_1 = np.zeros(len(article))
    gene_2 = np.zeros(len(article))
    gene_3 = np.zeros(len(article))
    gene_4 = np.zeros(len(article))
    gene_1[gene1] = 1
    gene_2[gene2] = 1
    gene_3[gene3] = 1
    gene_4[gene4] = 1

    mutate = np.random.randint(100)
    if mutate <80:
        #make mutate
        while True:
            print(str(itr) +' : mutation step')
            flag = False
            mut_index = np.random.randint(len(article))
            g_1_index = list(np.where(gene_1 == 1)[0])
            g_2_index = list(np.where(gene_2 == 1)[0])
            g_3_index = list(np.where(gene_3 == 1)[0])
            g_4_index = list(np.where(gene_4 == 1)[0])

            prev = len(g_1_index)
            mut_index = np.random.randint(len(article))
            if mut_index not in g_1_index:
                gene_1[mut_index] = 1
                rand_index = np.random.choice(g_1_index)
                gene_1[rand_index] = 0
                flag = True
            mut_index = np.random.randint(len(article))
            if mut_index not in g_2_index:
                gene_2[mut_index] = 1
                rand_index = np.random.choice(g_2_index)
                gene_2[rand_index] = 0
                flag = True
            mut_index = np.random.randint(len(article))
            if mut_index not in g_3_index:
                gene_3[mut_index] = 1
                rand_index = np.random.choice(g_3_index)
                gene_3[rand_index] = 0
                flag = True
            mut_index = np.random.randint(len(article))
            if mut_index not in g_4_index:
                gene_4[mut_index] = 1
                rand_index = np.random.choice(g_4_index)
                gene_4[rand_index] = 0

                flag = True
            after1 = len(list(np.where(gene_1 == 1)[0]))
            after2 = len(list(np.where(gene_2 == 1)[0]))
            after3 = len(list(np.where(gene_3 == 1)[0]))
            after4 = len(list(np.where(gene_4 == 1)[0]))
            print(after1)
            print(after2)
            print(after3)
            print(after4)
            if flag == True: break

    print('first gene result')
    for index in list(np.where(gene_1 == 1)[0]):
        print(article[index])

    print('#nsecond gene result')
    for index in list(np.where(gene_2==1)[0]):
        print(article[index])


    print('#nthird gene result')
    for index in list(np.where(gene_3==1)[0]):
        print(article[index])


    print('#nfourt gene result')
    for index in list(np.where(gene_4==1)[0]):
        print(article[index])

print('title : ' ,title)

for word in title.split():
    try:
        print('word : ',word , embedding_model.most_similar(positive=[word], topn=10))
    except:
        print(word)
#print(embedding_model.most_similar(positive=["연구진"], topn=10))
#print(embedding_model.most_similar(positive=["감염"], topn=10))

for itr, sent in enumerate(input_article[article_num]):
    print(itr, ' : ', sent)