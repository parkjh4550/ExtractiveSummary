# version6
# use only difference fitness function.

from gensim.models import Word2Vec
import numpy as np
import scipy.spatial.distance as Dist
from math import log
from sklearn.metrics.pairwise import cosine_similarity
embedding_model = Word2Vec.load('./data/word2vec/word2vec_model')
#print(embedding_model.most_similar(positive=["미국"], topn=10))

genes_num = 12
offsprint_num = 6       # even number -> makes 2 genes for each iteration
word_vec_size = 50
article_num = 10
summary_length =15
iteration_num = 200
NUM_SELECT = 6    # number of genes selected from child/ tournament

def gene_to_sentences(gene, article):
    sentence_list = []
    index = list(np.where(gene == 1)[0])
    for i in index:
        sentence_list.append(article[i])

    return sentence_list
def get_doc_vec(sentence_list):

    doc_vec = np.zeros(word_vec_size)
    for sentence in sentence_list:
        sen_vec = np.zeros(word_vec_size)
        tmp_s = sentence.split()
        for word in tmp_s:
            # print(word)
            try:
                word_vec = embedding_model.wv[word]
                sen_vec += word_vec
            except KeyError:
                cnt = 0

        sen_vec = sen_vec / len(tmp_s)
        doc_vec += sen_vec
    doc_vec = doc_vec / len(sentence_list)
    return doc_vec

def get_fitness(article, genes, title):

    # 2. Calc Diff of each summary
    diff_list = []
    print(' len : ', len(genes))
    for g1 in genes:
        tmp = []
        for g2 in genes:
            g1_sentences = gene_to_sentences(g1, article)
            g2_sentences = gene_to_sentences(g2, article)
            #print(g1_sentences)
            g1_doc = get_doc_vec(g1_sentences)
            g2_doc = get_doc_vec(g2_sentences)
            #print(g1_doc)
            diff = Dist.cosine(g1_doc, g2_doc)

            tmp.append(diff)


        diff_list.append(sum(tmp))

    # 2-1. Normalization
    diff_sum = sum(diff_list)
    tmp = []
    for diff in diff_list:
        print('diff_sum : ', diff_sum)
        print('diff : ', diff)
        if diff_sum != 0 :
            tmp.append(diff/diff_sum)
        elif diff_sum == 0:
            tmp.append(0)
    diff_list = tmp

    fit_list = []
    for itr in range(len(diff_list)):
        diff = diff_list[itr]
        fitness = diff
        print(fitness)
        fit_list.append(fitness)

    return fit_list

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

genes = []
for itr in range(genes_num):
    tmp_gene = np.zeros(len(article)).astype('int32')
    index = np.random.choice(len(article), summary_length, replace=False)
    tmp_gene[index] = int(1)
    genes.append(tmp_gene)


fitness_log = []
#start spreading
for step in range(iteration_num):

    #2. Calc Fitness
    fitness = []

    fitness = get_fitness(article, genes, title)

    # 3. Optimal Solution ??


    # 4. Selection
    # select best genes and save index
    sort_list = fitness[:]  # by adding [:] it has different memory address
    sort_list.sort(reverse=True)
    print('sorted fitness : ', sort_list)

    best_index = []
    selected_genes = []
    """
    for itr in range(NUM_SELECT):
        select_index = fitness.index((sort_list[itr]))
        best_index.append( select_index )
        selected_genes.append( genes[select_index])
    """
    if step != 0:
        for itr in range(len(sort_list)):
            select_index = fitness.index((sort_list[itr]))
            best_index.append(select_index)
            selected_genes.append(genes[select_index])

        index_interval = int(len(selected_genes) / NUM_SELECT)
        index_interval = 2
        selected_genes = selected_genes[0:len(selected_genes):index_interval]
    elif step == 0:
        selected_genes = genes

    # result check
    if step % 10 == 0:
        for i, fit in enumerate(fitness):
            print(str(step) + ' result')
            print(i, ' gene : ', fit)
            for index in list(np.where(genes[i] == 1)[0]):
                print(article[index])
        fitness_log.append(sort_list[:4])


    # 5. CrossOver
    print(str(step) + ' : cut choice')
    # if index is 0, then there is no left side genes.
    # so index  range : [1,2,3, ... max_len -1]
    # leftsize chrosome : 0, rightside chrosome : 1,2,3, .

    genes = []
    gene_range = range(NUM_SELECT)
    index_range = range(1, summary_length)
    for itr in range(int(offsprint_num/2)):      # each iteration, it makes 2 offsprings
        gene = np.random.choice(gene_range, 2, replace=False)
        print('selected gene: ', gene)
        cut = np.random.choice(index_range, 2, replace=False)
        cut.sort()  # to crossover appropriately, we have to sort the cut index

        # two points crossover
        # give part of gene2 into middle of gene1
        tmp_gene = np.hstack([selected_genes[gene[0]][:cut[0]], selected_genes[gene[1]][cut[0]:cut[1]], selected_genes[gene[0]][cut[1]:]])
        tmp_gene2 = np.hstack([selected_genes[gene[1]][:cut[0]], selected_genes[gene[0]][cut[0]:cut[1]], selected_genes[gene[1]][cut[1]:]])

        # one points crossover
        tmp_gene = np.hstack([selected_genes[gene[0]][:cut[0]], selected_genes[gene[0]][cut[0]:]])
        tmp_gene2 = np.hstack([selected_genes[gene[1]][:cut[1]], selected_genes[gene[1]][cut[1]:]])

        tmp_gene3 = np.hstack([selected_genes[gene[0]][:cut[1]], selected_genes[gene[0]][cut[1]:]])
        tmp_gene4 = np.hstack([selected_genes[gene[1]][:cut[0]], selected_genes[gene[1]][cut[0]:]])

        genes.append(tmp_gene)
        genes.append(tmp_gene2)
        genes.append(tmp_gene3)
        genes.append(tmp_gene4)

    # 6. Mutation
    # cal Mutation for each genes
    for itr, gene in enumerate(genes):
        mutate = np.random.randint(100) # mutate : 0 ~ 99
        #threshold = (1 - fitness[itr]) * 100  # if the fitness value is small, more mutation will be happened
        threshold = 20
        tmp_gene = gene
        if mutate < threshold:
            # make mutate
            flag = True

            while flag:
                print(str(step) + ' : mutation step')
                mut_index = np.random.randint(len(article))

                g_index = list(np.where(gene == 1)[0])  # get the location of 1's in the gene

                prev = len(g_index)
                mut_index = np.random.randint(len(article)) # select the chrosome that will be mutated

                if mut_index not in g_index:    # when the mutate index is already 1, it neglect the mutation
                    tmp_gene[mut_index] = 1     # mutate 0  -> 1
                    rand_index = np.random.choice(g_index)  # random select to make 1 -> 0
                    tmp_gene[rand_index] = 0
                    flag = False

                #if flag == True: break  # if the gene is mutated, break the while loop

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