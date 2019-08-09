# coding: utf-8
import os
import sys
import socket
import requests
import itertools
import re
import pymorphy2
import numpy as np
import pandas as pd
import json
import smtplib as smtp
from pandas.io.json import json_normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from urllib.parse import urlencode
from random import randint, getrandbits
from urllib.request import urlopen, URLError, Request
from bs4 import BeautifulSoup
from gensim.models.word2vec import Word2Vec
from tqdm import tqdm
from pymorphy2 import tokenizers
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_absolute_error
from time import sleep, gmtime, strftime
from flask import Flask, request, Response
from flask_cors import CORS
from threading import Thread

morph = pymorphy2.MorphAnalyzer()
TIMEOUT = 2
MAX_TRY = 2
MAX_LEN = 200
MIN_TIME_SLEEP = 1
MAX_TIME_SLEEP = 10
N_SPLITS_SPREAD = 1.5
DIR_LOGS = './logs'
CPU_NUM = 2

app = Flask(__name__)
CORS(app)
app.config['JSON_AS_ASCII'] = False

def get_html(url_page, timeout=TIMEOUT, max_try_count=MAX_TRY):
    socket.setdefaulttimeout(timeout * 10)
    flag = True
    count = 0
    html = ''
    while flag:
        try:
            html = urlopen(url_page, timeout=timeout)
            flag = False
        except URLError as e:
            print('URLError... ', e)
            count += 1
            if count <= max_try_count: 
                flag = True
            else:
                flag = False
        except socket.timeout as e:
            print('socket timeout... ', e)
            count += 1
            if count <= max_try_count: 
                flag = True
            else:
                flag = False
    return html
def get_links_stupid(query):
    result = []
    url = 'http://www.google.ru/search?q='
    html = requests.get(url + query + '&filter=0')
    soup = BeautifulSoup(html.text, 'html.parser')
    h3 = soup.find_all('h3', {'class': 'r'})
    result.extend([re.search('url\?q=(.*)&sa=', hi.findAll('a', href=True)[0].get('href')).group(1)
                   for hi in h3 
                   if 'http' in hi.findAll('a', href=True)[0].get('href')])
    return result
def get_text(link):
    html = get_html(link)
    soup = BeautifulSoup(html, 'html.parser')
    for script in soup(['script', 'style']):
        script.extract()
    page_text = soup.get_text()
    for ch in ['\n', '\t', '\r']:
        page_text = page_text.replace(ch, ' ')
    return ' '.join(page_text.split())
def preprocessing(sentence):
    s = re.sub('[^а-яА-Яa-zA-Z]+', ' ', sentence).strip().lower()
    s = re.sub('ё', 'е', s)
    funсtion_words = {'INTJ', 'PRCL', 'CONJ', 'PREP'}
    lemmatized_words = list(map(lambda word: morph.parse(word)[0], s.split()))
    result = []
    for word in lemmatized_words:
        if word.tag.POS not in funсtion_words:
            result.append(word.normal_form)
    return result
def word2vec_predict(sentence, word2vec_model):
    result = [word2vec_model[word] for word in sentence if word in word2vec_model.wv.vocab]
    if result:
        return list(np.mean(result, axis=0))
    else:
        return [0] * word2vec_model.vector_size
def get_vector_w2v(text, word2vec_model):
    vector_text = preprocessing(text)
    return word2vec_predict(vector_text, word2vec_model)
def search_text_range(total_len, max_len):
    if total_len - max_len > max_len:
        start = randint(max_len, total_len - max_len)
        end = start + max_len
        yield (start, end)
    elif total_len > max_len:
        yield (max_len, total_len)
    else:
        yield (0, max_len)
def search_text_range(total_len, max_len):
    start = randint(max_len, total_len - max_len)
    end = start + max_len - randint(0, 10)
    yield (start, end)
def get_search_links(text_main, n_splits, max_len, min_time_sleep, max_time_sleep):
    total_len = len(text_main)
    search_text_list = []
    search_text_list.append((0, max_len))
    for _ in range(n_splits - 1):
        search_text_list.append(next(search_text_range(total_len, max_len)))
    search_links = []
    for borders in tqdm(search_text_list):
        search_batch = text_main[borders[0]:borders[1]]
        search_links.extend(get_links_stupid(search_batch))
        sleep(randint(min_time_sleep, max_time_sleep))
    return search_links
def get_vector_tfidf(text_main, text_clone):
    tfidf_text = []
    tfidf_text.append(' '.join(preprocessing(text_main)))
    tfidf_text.append(' '.join(preprocessing(text_clone)))
    tfidf_vectorizer = TfidfVectorizer(ngram_range=(1,1), max_df=1.0, min_df=1)
    features = tfidf_vectorizer.fit_transform(tfidf_text)
    return pd.DataFrame(features.todense(), columns=tfidf_vectorizer.get_feature_names())
def get_vector_bow(text_main, text_clone):
    text_main = preprocessing(text_main)
    text_clone = preprocessing(text_clone)
    words_counts = set(text_main)
    words2index = {key: value for (key, value) in zip(sorted(words_counts), range(len(words_counts)))} 
    main_vector, clone_vector = np.zeros(len(words_counts)), np.zeros(len(words_counts))
    for x in text_main:
        try:
            main_vector[words2index[x]] += 1
        except:
            pass
    for x in text_clone:
        try:
            clone_vector[words2index[x]] += 1
        except:
            pass
    return main_vector, clone_vector
def get_json_clones(url_main, dest_email, query_id, max_len=MAX_LEN, min_time_sleep=MIN_TIME_SLEEP, max_time_sleep=MAX_TIME_SLEEP):
    try:
        print('url to search clones: ', url_main)
        text_main = get_text(url_main)
        total_len = len(text_main)
        n_splits = int(N_SPLITS_SPREAD * total_len // max_len)
        #n_splits = 2 #!!!TEST!!!
        #print('total length from main url: ', total_len, ' | number of splits: ', n_splits)
        word2vec_dir = './w2v_model'
        w2v_model = Word2Vec.load(word2vec_dir + '/' + 'w2v_mdl_3')
        print('w2v model loaded...')
        search_links = get_search_links(text_main, n_splits, max_len, min_time_sleep, max_time_sleep)
        print('links found: ', len(search_links))
        search_links = list(set(search_links))
        clean_list = [url_main, '.pdf', '.PDF', '.doc', '.DOC', '.docx', '.DOCX', '.xls', '.XLS', '.xlsx', '.XLSX']
        for elm in clean_list:
            search_links = [x for x in search_links if elm not in x]
        print('links to search (cleaned): ', len(search_links))
        #print('first links: ', search_links[:3]) #!!!TEST!!!
        list_clones = []
        vector_url_main = get_vector_w2v(text_main, w2v_model)
        for link in tqdm(search_links):
            try:
                text_clone = get_text(link)
                vector_link = get_vector_w2v(text_clone, w2v_model)
                df = get_vector_tfidf(text_main, text_clone)
                v0, v1 = get_vector_bow(text_main, text_clone)
                list_clones.append(
                    {'url': link,
                     'w2v_cosine': np.float64(cosine_similarity([vector_url_main], [vector_link])[0][0]), 
                     'tfidf_cosine': np.float64(cosine_similarity([df.iloc[0].values], [df.iloc[1].values])[0][0]),
                     'bow_cosine': np.float64(cosine_similarity([v0], [v1])[0][0]),
                     'bow_mae': np.float64(mean_absolute_error(v0, v1))}
                )
            except:
                print('failed to get similarity: ', link)
        print('total links processed: ', len(list_clones))
        dict_clones = {'url_main': url_main}
        dict_clones.update({'url_clones': list_clones})
    except:
        print('internal server error, check url to search clones')
        dict_clones = {'url_main': url_main}
        dict_clones.update({'status': 'internal server error, check url to search clones'})
    dir_logs = DIR_LOGS
    if not os.path.exists(dir_logs):
        os.makedirs(dir_logs)
    file_name = '{}/{}'.format(dir_logs, query_id)
    with open(file_name, 'w') as log_file:
        json.dump(dict_clones, log_file)
    print('log file created: ', file_name)
    error_mail = send_mail(dest_email, get_email_text(json.dumps(dict_clones)))
    if error_mail:
        print('email was not sent to: {} | error: {}'.format(dest_email, error_mail))
    else:
        print('email was sent to: ', dest_email)
def get_clones_logs(query_id):
    errors = []
    dir_logs = DIR_LOGS
    try:
        file_name = '{}/{}'.format(dir_logs, query_id)
        with open(file_name) as file:
            dict_clones = json.load(file)
    except:
        errors.append('could not load id {} data, not ready or does not exist'.format(query_id))
        dict_clones = []
    return (dict_clones, errors)
def get_email_text(json_clones):
    data = json.loads(json_clones)
    try:
        df = json_normalize(data, record_path=['url_clones'], meta=['url_main'])
        cols = ['url_main', 'url', 'bow_cosine', 'bow_mae', 'tfidf_cosine', 'w2v_cosine']
        df = df[cols].sort_values(['bow_cosine'], ascending=[False])[:10].reset_index()
        del df['index']
    except:
        df = json_normalize(data)
    return df.to_string()
def send_mail(dest_email, email_text):
    error = []
    try:
        email = 'app.notifications@yandex.ru'
        password = 'Notify2019'
        subject = 'Search clones notofication'
        message = 'From: {}\nTo: {}\nSubject: {}\n\n{}'.format(email, dest_email, subject, email_text)
        server = smtp.SMTP_SSL('smtp.yandex.com')
        server.login(email, password)
        server.auth_plain()
        server.sendmail(email, dest_email, message)
        server.quit()
    except smtp.SMTPException as e:
        error.append(e)
    return error
def to_json(data):
    return json.dumps(data)
def resp(code, data):
    return Response(status=code, mimetype='application/json', response=to_json(data))
def theme_validate():
    errors = []
    json = request.get_json()
    if json is None:
        errors.append('no JSON sent, check Content-Type header')
        return (None, errors)
    for field_name in ['url', 'email']:
        if type(json.get(field_name)) is not str:
            errors.append('field {} is missing or is not a string'.format(field_name))
    print('got json: ', json, ' | errors: ', errors)
    return (json, errors)
def theme_validate_id():
    errors = []
    json = request.get_json()
    if json is None:
        errors.append('no JSON sent, check Content-Type header')
        return (None, errors)
    for field_name in ['query_id']:
        if type(json.get(field_name)) is not str:
            errors.append('field {} is missing or is not a string'.format(field_name))
    print('got json: ', json, ' | errors: ', errors)
    return (json, errors)
        
@app.route('/search', methods=['POST'])
def search_clones_api():
    (json, errors) = theme_validate()
    if errors:
        return resp(400, {'errors': errors})
    query_id = '{}_{}'.format(strftime('%Y-%m-%d_%H-%M-%S', gmtime()), getrandbits(8))
    thread = Thread(target=get_json_clones, kwargs={'url_main': json['url'],
                                                    'dest_email': json['email'],
                                                    'query_id': query_id})
    thread.start()
    return resp(200, [{'status': 'search in progress'}, {'query_id': query_id}])
@app.route('/getclones', methods=['POST'])
def data_clones_api():
    (json, errors) = theme_validate_id()
    if errors:
        return resp(400, {'errors': errors})
    (json_clones, error_clones) = get_clones_logs(json['query_id'])
    if error_clones:
        return resp(500, {'error': error_clones})
    else:
        return resp(200, json_clones)    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000, debug=True)
