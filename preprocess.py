import json
import re
from konlpy.tag import Hannanum
from gensim.models import Word2Vec

"""
data format

{
	'title': "기사 제목",
	'source': "http://출처.출처",
	'slug': "can-be-used-as-unique-id",
	'length': 23,
	'summaries': [0, 1, 4, 7],
	'sentences': [
		"문장1",
		"문장2",
		...
	]
}
"""
# select mode for task
# 0 : write file
# 1 : divide a sentence with morpheme (using KoNLPy)
# 2 : make word2vec (using Gensim)
mode = 2

def load_data(flag):
    if flag == 0 :
        title_tmp = []
        input_tmp, target_tmp = [], []

        for i in range(1, 51):
            tmp_dict = {}
            if i < 10:
                file_name = '0' + str(i)
            else:
                file_name = str(i)
            print(file_name)
            with open('./data/original_data/' + file_name + '.json', 'r', encoding='UTF8') as f:
                data = json.load(f)

            title = data['title']
            content = data['sentences']
            summaries = data['summaries']

            title_tmp.append(title)
            input_tmp.append(content)
            target_tmp.append(summaries)

        return title_tmp, input_tmp, target_tmp

    elif flag == 1:
        data = []
        input_data = []
        title_data = []
        with open('./data/input_data.txt', 'r') as f:
            while True:
                line = f.readline()
                if not line: break
                line = line.split('\t')
                data.append(line)
                input_data.append(line)
        with open('./data/title_data.txt', 'r') as f:
            while True:
                line = f.readline()
                if not line: break
                line = line.split('\t')
                data.append(line)
                title_data.append(line)

        return data, input_data, title_data

    elif flag == 2:
        data = []
        with open('./data/morphemed_data.txt', 'r') as f:
            while True:
                line = f.readline()
                if not line: break
                line = line[:-1].split()
                data.append(line)

        return data


def remove_symbol(sentence):
    p = re.compile('\(.*?\)')
    tmp_s = re.sub(p, '', sentence)

    p = re.compile('\<.*?\>')
    tmp_s = re.sub(p, '', tmp_s)

    tmp_s = re.sub('[-=.#/!(\')?"":$“”‘’}\[\]]', '', tmp_s)
    if tmp_s[0] == ' ': tmp_s = tmp_s[1:]

    return tmp_s

def write_data(mode):
    title_data, input_data, target_data = load_data(flag=mode)
    with open('./data/input_data.txt', 'w') as f:
        for article in input_data:
            for itr,sentence in enumerate(article):
                if itr!= 0: f.write('\t')
                tmp_s = remove_symbol(sentence)
                f.write(tmp_s)
            f.write('\n')

    with open('./data/target_data.txt', 'w') as f:
        for article in target_data:
            for itr, target in enumerate(article):
                if itr!=0: f.write('\t')
                f.write(str(target))
            f.write('\n')

    with open('./data/title_data.txt', 'w') as f:
        for title in title_data:
            tmp = remove_symbol(title)
            f.write(tmp+'\n')

    tmp = []
    for article in input_data:
        for sentence in article:

            print('처리 전 : ', sentence)
            p = re.compile('\(.*?\)')
            tmp_s = re.sub(p, '', sentence)

            p = re.compile('\<.*?\>')
            tmp_s = re.sub(p, '', tmp_s)

            tmp_s = re.sub('[-=.#/!(\')?"":$“”‘’}\[\]]', '', tmp_s)
            if tmp_s[0] == ' ': tmp_s = tmp_s[1:]
            print('처리 후 : ', tmp_s)

            tmp_s = tmp_s.split()
            tmp.extend(tmp_s)
    tmp = set(tmp)
    print('num of words : ', len(list(tmp)))


def divide_with_morpheme(raw_data, total=1):
    data = []
    hannanum = Hannanum()
    if total == 1:
        for itr, article in enumerate(raw_data):
            for sentence in article:
                #print(sentence)
                if sentence != '':
                    pos_result = hannanum.morphs(sentence)
                    tmp = " ".join(pos_result)
                    data.append(tmp)
            print(str(itr)+ 'th article processed')
            print('last sentence : ' + tmp)
        return data
    elif total ==0 :
        for itr, article in enumerate(raw_data):
            tmp_data = []
            for sentence in article:
                #print(sentence)
                if sentence != '':
                    pos_result = hannanum.morphs(sentence)
                    tmp = " ".join(pos_result)
                    tmp_data.append(tmp)
            print(str(itr)+ 'th article processed')
            print('last sentence : ' + tmp)
            data.append(tmp_data)
        return data

if mode == 0:
    # remove special symbols
    write_data(mode=mode)
elif mode == 1:
    # divide sentence into morphemes
    total_data, input_data, title_data = load_data(flag=mode)

    divided_data = divide_with_morpheme(total_data, total=1)
    divided_input = divide_with_morpheme(input_data, total=0)
    divided_title = divide_with_morpheme(title_data, total=0)

    with open('./data/morpheme/morphemed_data.txt', 'w') as f:
        for line in divided_data:
            print(line)
            f.write(line+'\n')

    with open('./data/morpheme/morphemed_input.txt', 'w') as f:
        for article in divided_input:
            for itr, sentence in enumerate(article):
                if itr !=0: f.write('\t')
                f.write(sentence)
            f.write('\n')

    with open('./data/morpheme/morphemed_title.txt', 'w') as f:
        for article in divided_title:
            for itr, sentence in enumerate(article):
                if itr != 0: f.write('\t')
                f.write(sentence)
            f.write('\n')

elif mode == 2:
    # train for word2vec
    total_data = []
    with open('./data/morpheme/morphemed_data.txt', 'r') as f:
        while True:
            line = f.readline()
            if not line: break
            line = line[:-1].split()
            total_data.append(line)
    embedding_model = Word2Vec(total_data, size=50, window=2, min_count=1, iter=100, sg=1)
    embedding_model.save('./data/word2vec/word2vec_model')
    print(embedding_model.most_similar(positive=["미국"], topn=10))


