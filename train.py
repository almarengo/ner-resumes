############################################  NOTE  ########################################################
#
#           Creates NER training data in Spacy format from JSON downloaded from Dataturks.
#
#           Outputs the Spacy training data which can be used for Spacy training.
#
############################################################################################################
import json
import random
import logging
from itertools import compress
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from spacy.gold import GoldParse
from spacy.scorer import Scorer
from sklearn.metrics import accuracy_score
def convert_dataturks_to_spacy(dataturks_JSON_FilePath):
    try:
        training_data = []
        lines=[]
        with open(dataturks_JSON_FilePath, 'r', encoding="utf8") as f:
            lines = f.readlines()

        for line in lines:
            data = json.loads(line)
            text = data['content']
            entities = []
            for annotation in data['annotation']:
                #only a single point in text annotation.
                point = annotation['points'][0]
                labels = annotation['label']
                # handle both list of labels or a single label.
                if not isinstance(labels, list):
                    labels = [labels]

                for label in labels:
                    #dataturks indices are both inclusive [start, end] but spacy is not [start, end)
                    entities.append((point['start'], point['end'] + 1 ,label))


            training_data.append((text, {"entities" : entities}))

        return training_data
    except Exception as e:
        logging.exception("Unable to process " + dataturks_JSON_FilePath + "\n" + "error = " + str(e))
        return None


import bisect
import sys

class Overlap():

    def __init__(self):
        self._intervals = []

    def intervals(self):
        return self._intervals

    def put(self, interval):
        istart, iend = interval
        # Ignoring intervals that start after the window.                                       
        i = bisect.bisect_right(self._intervals, (iend, sys.maxsize))

        # Look at remaining intervals to find overlap.                                          
        for start, end in self._intervals[:i]:
            if end > istart:
                return False
        bisect.insort(self._intervals, interval)
        return True

import spacy
################### Train Spacy NER.###########
def train_spacy():

    TRAIN_DATA = convert_dataturks_to_spacy("traindata.json")
    nlp = spacy.blank('en')  # create blank Language class
    # create the built-in pipeline components and add them to the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if 'ner' not in nlp.pipe_names:
        ner = nlp.create_pipe('ner')
        nlp.add_pipe(ner, last=True)

    ov = Overlap()
    idx = 0
    for text, annotations in TRAIN_DATA:
        TRAIN_DATA[idx][1]['entities'] = list(dict.fromkeys(TRAIN_DATA[idx][1]['entities']))
        input_list = annotations.get('entities')
        input_list = sorted(input_list)
        yz_list = [(start, end) for start, end, _ in input_list]
        for i in yz_list:
            ov.put(i)
        keep = []
        for entity in annotations.get('entities'):
            for interval in ov.intervals():
                if ((entity[0] == interval[0]) & (entity[1] == interval[1])):
                    keep.append(True)
                    break
            else:
                keep.append(False)
        TRAIN_DATA[idx][1]['entities'] = list(compress(annotations.get('entities'), keep))
        idx += 1   

    # add labels
    for _, annotations in TRAIN_DATA:
         for ent in annotations.get('entities'):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):  # only train NER
        optimizer = nlp.begin_training()
        for itn in range(20):
            print("Statring iteration " + str(itn))
            random.shuffle(TRAIN_DATA)
            losses = {}
            for text, annotations in TRAIN_DATA:
                nlp.update(
                    [text],  # batch of texts
                    [annotations],  # batch of annotations
                    drop=0.2,  # dropout - make it harder to memorise data
                    sgd=optimizer,  # callable to update weights
                    losses=losses)
            print(losses)
    #test the model and evaluate it
    examples = convert_dataturks_to_spacy("testdata.json")
    tp=0
    tr=0
    tf=0

    ta=0
    c=0    

    idx = 0
    for text, annotations in examples:
        examples[idx][1]['entities'] = list(dict.fromkeys(examples[idx][1]['entities']))
        input_list = annotations.get('entities')
        input_list = sorted(input_list)
        yz_list = [(start, end) for start, end, _ in input_list]
        for i in yz_list:
            ov.put(i)
        keep = []
        for entity in annotations.get('entities'):
            for interval in ov.intervals():
                if ((entity[0] == interval[0]) & (entity[1] == interval[1])):
                    keep.append(True)
                    break
            else:
                keep.append(False)
        examples[idx][1]['entities'] = list(compress(annotations.get('entities'), keep))
        idx += 1 

    d={}
    for text,annot in examples:

        doc_to_test=nlp(text)
        
        for ent in doc_to_test.ents:
            if ent.label_ in d.keys():
                continue
            else:
                d[ent.label_]=[0,0,0,0,0,0]
        for ent in doc_to_test.ents:
            doc_gold_text= nlp.make_doc(text)
            gold = GoldParse(doc_gold_text, entities=annot.get("entities"))
            y_true = [ent.label_ if ent.label_ in x else 'Not '+ent.label_ for x in gold.ner]
            y_pred = [x.ent_type_ if x.ent_type_ ==ent.label_ else 'Not '+ent.label_ for x in doc_to_test]  
            if(d[ent.label_][0]==0):
                (p,r,f,s)= precision_recall_fscore_support(y_true,y_pred,average='weighted')
                a=accuracy_score(y_true,y_pred)
                d[ent.label_][0]=1
                d[ent.label_][1]+=p
                d[ent.label_][2]+=r
                d[ent.label_][3]+=f
                d[ent.label_][4]+=a
                d[ent.label_][5]+=1
        c+=1
    for i in d:
        print("\n For Entity "+i+"\n")
        print("Accuracy : "+str((d[i][4]/d[i][5])*100)+"%")
        print("Precision : "+str(d[i][1]/d[i][5]))
        print("Recall : "+str(d[i][2]/d[i][5]))
        print("F-score : "+str(d[i][3]/d[i][5]))

        
train_spacy()
