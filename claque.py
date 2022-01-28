from nltk.stem import PorterStemmer
import os
import csv
import json
import spacy
import pickle

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import tensorflow as tf
from sklearn import preprocessing, model_selection
from sklearn.metrics import classification_report,confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM,Dense,Embedding,Dropout
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

import keras_tuner as kt
from tensorflow import keras


class ClassifyQuestion:
    def __init__(self, is_binary):
        self.is_binary = is_binary
        self.ps = PorterStemmer()
        self.sp = spacy.load('en_core_web_sm')
        self.categories = {self.ps.stem("name"):"A1", "list":"A1", self.ps.stem("define"):"A1", self.ps.stem("label"):"A1",
                    self.ps.stem("restate"):"A2", self.ps.stem("order"):"A2",
                    self.ps.stem("state"):"A3", self.ps.stem("determine"):"A3",
                    self.ps.stem("distinguish"):"A4", self.ps.stem("classify"):"A4",
                    self.ps.stem("select"):"A5", self.ps.stem("prioritize"):"A5",

                    self.ps.stem("identify"):"B1", self.ps.stem("locate"):"B1",
                    self.ps.stem("describe"):"B2", self.ps.stem("explain"):"B2",
                    self.ps.stem("illustrate"):"B3", self.ps.stem("show"):"B3",
                    self.ps.stem("examine"):"B3", self.ps.stem("analyze"):"B3",
                    self.ps.stem("rank"):"B3", self.ps.stem("compare"):"B3",

                    self.ps.stem("tell"):"C1", self.ps.stem("describe"):"C1",
                    self.ps.stem("summarize"):"C2", self.ps.stem("translate"):"C2",
                    self.ps.stem("solve"):"C3", self.ps.stem("demonstrate"):"C3",
                    self.ps.stem("deduct"):"C4", self.ps.stem("diagram"):"C4",
                    self.ps.stem("conclude"):"C5", self.ps.stem("choose"):"C5",

                    self.ps.stem("interpret"):"D2", self.ps.stem("paraphrase"):"D2",
                    self.ps.stem("find out"):"D3", self.ps.stem("use"):"D3",
                    self.ps.stem("infer"):"D4", self.ps.stem("examine"):"D4",
                    self.ps.stem("judge"):"D5", self.ps.stem("justify"):"D5",}
        self.binary = {
            "A1":'EASY',
            "A2":'EASY',
            "A3":'EASY',
            "A4":'EASY',
            "A5":'EASY',
            "B1":'EASY',
            "B2":'DIFFICULT',
            "B3":'DIFFICULT',
            "B4":'DIFFICULT',
            "B5":'DIFFICULT',
            "C1":'DIFFICULT',
            "C2":'DIFFICULT',
            "C3":'DIFFICULT',
            "C4":'DIFFICULT',
            "C5":'DIFFICULT',
            "D1":'DIFFICULT',
            "D2":'DIFFICULT',
            "D3":'DIFFICULT',
            "D4":'DIFFICULT',
            "D5":'DIFFICULT',
        }
        self.bloom_keys = list(self.categories.keys())
        self.punct = '?,.;'


    def bloom_categorize(self, question, csvfile, nocatfile):
        stemmed_words = [self.ps.stem(word) for word in question.split()]
        words = question.split()
        sen = self.sp(question)
        is_bloom_question = False
        for i in range(len(words)):
            if sen[i].pos_ == 'VERB' and stemmed_words[i] in self.bloom_keys:
                label = self.categories[stemmed_words[i]]
                csvfile.write(f'{self.clean_word(question)};{label}\n')
                is_bloom_question = True
                break
        if not is_bloom_question:
            nocatfile.write(f'{self.clean_word(question)};{"<UNK>"}')

    def bloom_categorize_binary(self, question, csvfile, nocatfile):
        stemmed_words = [self.ps.stem(word) for word in question.split()]
        words = question.split()
        sen = self.sp(question)
        is_bloom_question = False
        for i in range(len(words)):
            if sen[i].pos_ == 'VERB' and stemmed_words[i] in self.bloom_keys:
                binary_label = self.binarize(self.categories[stemmed_words[i]])
                csvfile.write(f'{self.clean_word(question)};{binary_label}\n')
                is_bloom_question = True
                break
    
    def binarize(self, label):
        return self.binary[label]

    def categorize_hotpotqa(self, filepath):
        with open(filepath, 'r') as f:
            newfile = json.load(f)
            
        with open('qc_hotpot_results.csv', 'w', encoding='utf-8') as csvfile:
            with open('qc_hotpot_noclass.csv', 'w', encoding='utf-8') as nocatfile:
                for line in newfile:
                    question = line['question']
                    self.bloom_categorize(question, csvfile, nocatfile)
    
    def categorize_arc_dir(self, dirpath):
        for root, dirs, files in os.walk(dirpath):
            for file in files:
                if file.endswith('.csv'):
                    path = root + '/' + file
                    self.categorize_arc(path)
                    
    def categorize_squad_dir(self, dirpath):
        for root, dirs, files in os.walk(dirpath):
            for file in files:
                if file.endswith('.json'):
                    path = root + '/' + file
                    self.categorize_squad(path)

    def categorize_arc_binary(self, filepath):
        print("categorize...")
        with open(filepath, 'r') as f:
            newfile = csv.reader(f)
            _ = next(newfile)
            with open('qc_results.csv', 'a', encoding='utf-8') as csvfile:
                with open('qc_arc_noclass.csv', 'w', encoding='utf-8') as nocatfile:
                    for line in newfile:
                        question = line[9].split('(')[0].strip() # extracts the question
                        self.bloom_categorize_binary(question, csvfile, nocatfile)
        print("Saved.")
        
    def categorize_arc(self, filepath):
        print("categorize...")
        with open(filepath, 'r') as f:
            newfile = csv.reader(f)
            _ = next(newfile)
            with open('qc_results.csv', 'a', encoding='utf-8') as csvfile:
                with open('qc_arc_noclass.csv', 'w', encoding='utf-8') as nocatfile:
                    for line in newfile:
                        question = line[9].split('(')[0].strip() # extracts the question
                        self.bloom_categorize(question, csvfile, nocatfile)
        print("Saved.")

    def categorize_squad_binary(self, filepath):
        print("categorize...")
        with open(filepath, 'r') as f:
            newfile = json.load(f)

        with open('qc_results.csv', 'a', encoding='utf-8') as csvfile:
            with open('qc_squad_noclass.csv', 'w', encoding='utf-8') as nocatfile:
                for line in newfile['data']:
                    for para in line['paragraphs']:
                        for q in para['qas']:
                            question = q['question']
                            self.bloom_categorize_binary(question, csvfile, nocatfile)
        print("Saved.")

    def categorize_squad(self, filepath):
        print("categorize...")
        with open(filepath, 'r') as f:
            newfile = json.load(f)

        with open('qc_results.csv', 'a', encoding='utf-8') as csvfile:
            with open('qc_squad_noclass.csv', 'w', encoding='utf-8') as nocatfile:
                for line in newfile['data']:
                    for para in line['paragraphs']:
                        for q in para['qas']:
                            question = q['question']
                            self.bloom_categorize(question, csvfile, nocatfile)
        print("Saved.")

    def clean_word(self, word):
        for char in self.punct:
            if char in word:
               word = word.replace(char,'') 
        return word

    def get_complexity(self, question):
        with open('./res/params.pkl', 'rb') as f:
            char_to_ind, mapping = pickle.load(f)
        qm = QuestionMasker()

        masked_question = qm.mask_question(question)
        loaded_model = tf.keras.models.load_model('./res/complexity-model.h5')
        raw_inputs = []

        for elem in masked_question.split(" "):
            raw_inputs.append(char_to_ind[elem])
        raw_inputs = np.array(raw_inputs)

        result = np.zeros(26)
        result[:raw_inputs.shape[0]] = raw_inputs[:26]

        result = result.reshape(1,26)

        y_prob = loaded_model.predict(result)
        y_classes = y_prob.argmax(axis=-1)
        return mapping[y_classes[0]]


class QuestionMasker:
    def __init__(self):
        print("Mask Questions with POS Tags")
        self.sp = spacy.load('en_core_web_sm')
        self.punct = '?,.;'
        self.qwords = ['what', 'who', 'how', 'when', 'where', 'why', 'which', 'whom', 'whose']
        self.easy = 0
        self.difficult = 0

    # retrieves the POS-Tags of a questions and returns them space-separated in a string
    def mask_question(self, sentence):
        sen_array = sentence.split()
        sen = self.sp(sentence)

        results = []
        for i in range(len(sen_array)):
            if sen_array[i].lower() in self.qwords:
                results.append(sen_array[i].upper())
            else:
                results.append(sen[i].pos_)
        return " ".join(results)
    
    def combine_question(self, sentence):
        sen_array = sentence.split()
        sen = self.sp(sentence)
        results = []
        for e, word in enumerate(sen_array):
            word = self.clean_word(word)
            if word.lower() in self.qwords:
                results.append(word.upper())
            else:
                results.append(word.lower())
        for e, word in enumerate(sen_array):
            word = self.clean_word(word)
            if word.lower() in self.qwords:
                results.append(word.upper())
            else:
                results.append(sen[e].pos_)
        return " ".join(results)

    def tuple_question(self, sentence):
        sen_array = sentence.split()
        sen = self.sp(sentence)

        results = []
        for e, word in enumerate(sen_array):
            word = self.clean_word(word)
            if word.lower() in self.qwords:
                results.append(word.upper())
            else:
                results.append(word+' '+sen[e].pos_)
        return "\t".join(results)
    
    def clean_word(self, word):
        for char in self.punct:
            if char in word:
               word = word.replace(char,'') 
        return word
                
    # masks all questions in a csv file, when the label is included
    def mask_file_tupled(self, filepath):
        # infile
        outpath = 'masked_' + filepath
        with open(filepath, 'r') as f:
            csvreader = csv.reader(f, delimiter=';')

            # outfile
            with open(outpath, 'w', encoding='utf-8') as csvfile:
                csvfile.write('text,label\n')
                for line in csvreader:
                    #print("Line", line)
                    if line[1] == 'DIFFICULT':
                        self.difficult += 1
                    else:
                        self.easy += 1
                    q_masked = self.tuple_question(line[0])
                    csvfile.write(f'{q_masked},{line[1]}\n')

    def mask_file(self, inpath, outpath):
        with open(inpath, 'r') as f:
            csvreader = csv.reader(f, delimiter=';')

            # outfile
            with open(outpath, 'w', encoding='utf-8') as csvfile:
                csvfile.write('text,label\n')
                for line in csvreader:
                    #print("Line", line)
                    if line[1] == 'DIFFICULT':
                        self.difficult += 1
                    else:
                        self.easy += 1
                    q_masked = self.mask_question(line[0])
                    csvfile.write(f'{q_masked},{line[1]}\n')


class Training():
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("-v", "--verbose", help="print verbose output", action="store_true")
        parser.add_argument("-s", "--save", help="save model", action="store_true")
        self.args = parser.parse_args()
    
    def verboseprint(self, msg):
        if self.args.verbose:
            print(">>>", msg)

    def flatten(self, t):
        return [item for sublist in t for item in sublist]

    def data_from_file(self, path_to_file):
        self.verboseprint("Reading file...")
        with open(path_to_file, 'r', encoding='utf-8') as f:
            rawtext = f.read()

        lines = rawtext.splitlines()
        vocab = [line[:-3].split(" ") for line in lines]

        X, y = self.get_x_y(lines, vocab)
        X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.33, random_state=42)
        return X_train, X_test, y_train, y_test
    
    def get_x_y(self, lines, vocab):
        self.verboseprint("Formatting X and y...")
        y_train = self.flatten([line[-2:].split(" ") for line in lines])
        char_to_ind = {u:i for i, u in enumerate(set(self.flatten(vocab)))}

        ind_to_char = np.array(vocab, dtype=object)
        raw_inputs = []

        for elem in vocab:
            result = []
            for word in elem:
                result.append(char_to_ind[word])
            result = np.array(result)
            raw_inputs.append(result)

        X_train = tf.keras.preprocessing.sequence.pad_sequences(raw_inputs, padding="post")

        le = preprocessing.LabelEncoder()
        labels = le.fit_transform(y_train)
        
        y_cat_train = to_categorical(labels)
        inverse = le.inverse_transform(labels)
        mapping = {}

        for i in range(len(y_train)):
            index = np.where(y_cat_train[i]==1)
            mapping[index[0].astype(int)[0]] = inverse[i]

        with open('./res/params.pkl', 'wb') as f:
            pickle.dump([char_to_ind, mapping], f)

        return X_train, y_cat_train
    
    def model_builder(self, hp):
        model = Sequential()
        
        # Tune the number of units in the first Dense layer
        # Choose an optimal value between 32-512
        hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
        model.add(Dense(units=hp_units, activation='relu'))
        model.add(Dense(14))

        # Tune the learning rate for the optimizer
        # Choose an optimal value from 0.01, 0.001, or 0.0001
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate), loss=categorical_crossentropy, metrics=['accuracy']) 

        return model
    
    def create_model(self, x_shape, num_classes, rnn_neurons, X_train, y_train):
        self.verboseprint("Creating model...")
        model = Sequential()
        model.add(Dense(128, activation='relu', input_shape = (x_shape,)))
        model.add(Dense(64, activation='relu'))   

        model.add(Dense(num_classes, activation='softmax'))
        model.compile(optimizer='adam', loss=categorical_crossentropy, metrics=['accuracy']) 
        self.verboseprint("Model compiled successfully.")
        
        tuner = kt.Hyperband(self.model_builder,
                     objective='val_accuracy',
                     max_epochs=10,
                     factor=3,
                     directory='my_dir',
                     project_name='intro_to_kt')
        
        stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        
        tuner.search(X_train, y_train, epochs=50, validation_split=0.2, callbacks=[stop_early])

        # Get the optimal hyperparameters
        best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

        print(f"""
        The hyperparameter search is complete. The optimal number of units in the first densely-connected
        layer is {best_hps.get('units')} and the optimal learning rate for the optimizer
        is {best_hps.get('learning_rate')}.
        """)


        return model

    def train_model(self, X_train, X_test, y_cat_train, y_test):
        self.verboseprint("Training model...")
        early_stop = EarlyStopping(monitor='val_loss',patience=4)
        model = self.create_model(
                        x_shape=X_train.shape[1],
                        num_classes=y_cat_train.shape[1],
                        rnn_neurons=1026,
                        X_train = X_train,
                        y_train = y_cat_train)
        if self.args.save:
            model.save('./res/complexity-model.h5')
        self.verboseprint(model.summary())
        history = model.fit(X_train,y_cat_train,epochs=20, validation_data=(X_test, y_test),callbacks=[early_stop])
        return model

    def plot_results(self, history_dict, metric):
        self.verboseprint("Plotting results...")
        epochs = range(1, len(history_dict['loss']) + 1)

        plt.title(f'Training and validation {metric}')
        plt.xlabel('epochs')
        plt.ylabel(metric)

        plt.plot(epochs, history_dict[metric], 'ro', label=f"Training {metric}")
        plt.plot(epochs, history_dict[f'val_{metric}'], 'r', label=f"Validation {metric}")

        plt.legend()
        plt.show()

    def evaluate(self, X_test, y_test):
        with open('./res/params.pkl', 'rb') as f:
            char_to_ind, mapping = pickle.load(f)
        model = tf.keras.models.load_model('./res/complexity-model.h5')
        y_test_cat = [mapping[int(np.argwhere(y==1)[0])] for y in y_test]

        res = [[key for key, val in mapping.items() if val == elem] for elem in y_test_cat]
        res = self.flatten(res)

        y_prob = model.predict(X_test)
        predictions = y_prob.argmax(axis=-1)
        print(classification_report(res,predictions))

       
        plt.figure(figsize=(10,6))
        sns.heatmap(confusion_matrix(res,predictions),annot=True)

        plt.show()


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--binary', type= lambda x: x in ['YES', 'yes', '1', 'true', 'TRUE'], default=False)
    parser.add_argument('--lexic', type= lambda x: x in ['YES', 'yes', '1', 'true', 'TRUE'], default=False)
    args = parser.parse_args()
    
    if os.path.exists('./qc_results.csv'):
        os.remove('./qc_results.csv')
    if os.path.exists('./qc_arc_noclass.csv'):
        os.remove('./qc_arc_noclass.csv')
    if os.path.exists('./qc_squad_noclass.csv'):
        os.remove('./qc_squad_noclass.csv')
    if os.path.exists('./masked_qc_results.csv'):
        os.remove('./masked_qc_results.csv')

    #cq = ClassifyQuestion()
    #cq.categorize_squad('../Datasets/SQuAD/train-v2.0.json')
    
    qm = QuestionMasker()
    qc = ClassifyQuestion(is_binary=args.binary)
    
    arc_filepath = './arc_data'
    squad_filepath = './squad_data'
    if args.binary:
        qc.categorize_arc_dir_binary(arc_filepath)
        qc.categorize_squad_dir_binary(squad_filepath)       
    else: 
        qc.categorize_arc_dir(arc_filepath)
        qc.categorize_squad_dir(squad_filepath)
    resultfilepath = 'qc_results.csv'
    maskedresultfilepath = 'masked_qc_results.csv'

    if args.lexic:
        qm.mask_file_tupled(resultfilepath)
    else:
        qm.mask_file(resultfilepath,maskedresultfilepath)
        
    print('easy: ',qm.easy)
    print('difficult', qm.difficult)
    print("Done.")