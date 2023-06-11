import os
import py_vncorenlp
import numpy as np
import tensorflow as tf
import keras
import tensorflow_addons as tfa
from vncorenlp import VnCoreNLP

from transformers import PhobertTokenizer, TFAutoModel
from tensorflow.keras.layers import GlobalAveragePooling1D
from constant import *

CURRENT_DIR = os.getcwd()

class Data_Processor:
    def __init__(self):
        self.load_annotator()
        tokenizer_dir = os.path.join(CURRENT_DIR, TOKENIZER_PATH)
        self.tokenizer = PhobertTokenizer.from_pretrained("vinai/phobert-base", cache_dir=tokenizer_dir)

    def load_annotator(self):       
        vncorenlp_dir = os.path.join(CURRENT_DIR, VNCORENLP_PATH) 
        if not os.path.exists(vncorenlp_dir):
            os.makedirs(vncorenlp_dir)
            py_vncorenlp.download_model(save_dir=VNCORENLP_PATH)
        vncorenlp_dir = 'resource/vncorenlp/VnCoreNLP-1.2.jar'
        self.annotator = VnCoreNLP(vncorenlp_dir, annotators='wseg')
    
    def batch_annonate(self, text):
        def annotate(annotator):
            def apply(x):
                return ' '.join([' '.join(text) for text in annotator.tokenize(x)])
            return apply
        return list(map(annotate(self.annotator), text))
    
    def batch_tokenize(self, batch_input):
        tokenized_text = self.tokenizer(batch_input, padding ='max_length',return_tensors = 'np', truncation=True)
        return tokenized_text
    
    def batch_clean(self, batch_input):
        def clean_text(text):
            return text
        return list(map(clean_text, batch_input))
    
    def data_processing(self, batch_input):
        # batch_input = self.batch_clean(batch_input)
        batch_input = self.batch_annonate(batch_input)
        batch_input = self.batch_tokenize(batch_input)
        return batch_input

class Intent_Classifier:
    def __init__(self, intent_list, useLM=True):
        self.intent_list = intent_list
        self.load_model()
        self.useLM = useLM
        if useLM:
            self.load_lm()

    def load_model(self):
        model_dir = os.path.join(CURRENT_DIR, INTENT_MODEL_PATH)
        self.model = keras.models.load_model(model_dir)

    def load_lm(self):
        self.lm = TFAutoModel.from_pretrained("vinai/phobert-base").roberta
        self.pooling = GlobalAveragePooling1D()

    def predict(self, tokenized_input):
        if self.useLM:
            embed_data = self.lm(tokenized_input)[0]
            pooling_data = self.pooling(embed_data)
            output = self.model(pooling_data)[0]
        else:
            tokenized_input = tokenized_input['input_ids']
            tokenized_input[tokenized_input<3] = 0
            output = self.model(tokenized_input)[0]

        index = np.argmax(output)

        if output[index] < INTENT_THRESHOLD:
            intent = None
        else:
            intent = self.intent_list[index]
    
        confidence = round(output[index].numpy(), 4)

        return {
            'intent': intent,
            'confidence': confidence
        }


class Entity_Recognizer:
    def __init__(self, entity_list, useLM=False):
        self.entity_list = entity_list
        self.bio_entity_list = self.create_BIO_tagging(self.entity_list)
        self.tag2id, self.id2tag = self.create_entity_id_dict(self.bio_entity_list)
        self.useLM = useLM
        self.load_model()

    def load_model(self):
        model_dir = os.path.join(CURRENT_DIR, ENTITY_MODEL_PATH)
        self.model = keras.models.load_model(model_dir)

    def predict(self, tokenized_input):
        if not self.useLM:
            tokenized_input = tokenized_input['input_ids']
            tokenized_input[tokenized_input<3] = 0
        decoded_sequence, potentials, _, _ = self.model(tokenized_input[:, 1:])
        decoded_sequence = tf.squeeze(decoded_sequence).numpy()
        index = np.where((decoded_sequence!=0)&(decoded_sequence!=1))

        confidence_list = tf.nn.softmax(tf.squeeze(potentials), axis=-1).numpy()[index].max(axis=-1).tolist()
        bio_entity = [self.id2tag[i] for i in decoded_sequence[index]]
        value = tokenized_input[0, 1:][index]

        return {
            'bio_entity': bio_entity,
            'confidence': confidence_list,
            'value': value
        }

    def create_BIO_tagging(self, entity_list):
        return ['O'] + np.array([['B-'+e.upper(), 'I-'+e.upper()] for e in entity_list]).ravel().tolist()
    
    def create_entity_id_dict(self, bio_entity_list):
        return {e: bio_entity_list.index(e)+1 for e in bio_entity_list}, {bio_entity_list.index(e)+1: e for e in bio_entity_list}
    
    def create_entity_dict(self, bio_entity, decoded_value, entity_confidence):
        zipped_list = zip(bio_entity, decoded_value, entity_confidence)
        entity_dict = {e: [] for e in self.entity_list}
        for bio_e, v, c in zipped_list:
            bio_tag = bio_e.split('-')
            if bio_tag[0] == 'B':
                entity_dict[bio_tag[1].lower()].append([v, [round(c, 4)]])
            else:
                entity_dict[bio_tag[1].lower()][-1][0] += (' ' + v)
                entity_dict[bio_tag[1].lower()][-1][1].append(round(c, 4))

        for key in entity_dict.keys():
            if len(entity_dict[key]) != 0:
                for i, _ in enumerate(entity_dict[key]):
                    confidence_list = entity_dict[key][i][-1]
                    entity_dict[key][i][-1] = sum(confidence_list) / len(confidence_list)

        return entity_dict