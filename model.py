import os
import py_vncorenlp
import numpy as np
import tensorflow as tf
import keras
import tensorflow_addons as tfa
from vncorenlp import VnCoreNLP

from transformers import PhobertTokenizer
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
        # self.annotator = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=vncorenlp_dir)
        vncorenlp_dir = 'resource/vncorenlp/VnCoreNLP-1.2.jar'
        self.annotator = VnCoreNLP(vncorenlp_dir, annotators='wseg')
    
    def batch_annonate(self, text):
        def annotate(annotator):
            def apply(x):
                return ' '.join([' '.join(text) for text in annotator.tokenize(x)])
                # return ' '.join(annotator.word_segment(x))
            return apply
        return list(map(annotate(self.annotator), text))
    
    def batch_tokenize(self, batch_input):
        tokenized_text = self.tokenizer(batch_input, padding ='max_length',return_tensors = 'np', truncation=True)['input_ids']
        tokenized_text[tokenized_text<3] = 0
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
    def __init__(self, intent_list):
        self.intent_list = intent_list
        self.load_model()

    def load_model(self):
        model_dir = os.path.join(CURRENT_DIR, INTENT_MODEL_PATH)
        self.model = keras.models.load_model(model_dir)

    def predict(self, tokenized_input):
        output = self.model(tokenized_input)[0]
        index = np.argmax(output)

        if output[index] < INTENT_THRESHOLD:
            intent = None
        else:
            intent = self.intent_list[index]
        return {
            'intent': intent,
            'confidence': output[index]
        }


class Entity_Recognizer:
    def __init__(self, entity_list):
        self.entity_list = entity_list
        self.bio_entity_list = self.create_BIO_tagging(self.entity_list)
        self.tag2id, self.id2tag = self.create_entity_id_dict(self.bio_entity_list)

        self.load_model()

    def load_model(self):
        model_dir = os.path.join(CURRENT_DIR, ENTITY_MODEL_PATH)
        self.model = keras.models.load_model(model_dir)

    def predict(self, tokenized_input):
        decoded_sequence, potentials, _, _ = self.model(tokenized_input[:, 1:])
        decoded_sequence = decoded_sequence[0].numpy()
        potentials = potentials[0].numpy()
        index = np.where((decoded_sequence!=0)&(decoded_sequence!=1))
        max_confidence_index = np.unravel_index(np.argmax(potentials, axis=1), potentials.shape)
        max_confidence_index = np.ravel_multi_index(max_confidence_index, potentials.shape)
        confidence_list = potentials[max_confidence_index][index]
        if len(confidence_list) == 0:
            confidence_list = [1.]
        entity_list = [self.id2tag[i] for i in decoded_sequence[index]]
        value = tokenized_input[0, 1:][index]
        return {
            'entity': entity_list,
            # 'confidence': confidence_list,
            'value': value
        }

    def create_BIO_tagging(self, entity_list):
        return ['O'] + np.array([['B-'+e.upper(), 'I-'+e.upper()] for e in entity_list]).ravel().tolist()
    
    def create_entity_id_dict(self, bio_entity_list):
        return {e: bio_entity_list.index(e)+1 for e in bio_entity_list}, {bio_entity_list.index(e)+1: e for e in bio_entity_list}
    
    def create_entity_dict(self, entity_list, decoded_value):
        role = ''
        if 'B-ROLE' in entity_list:
            B_role_index = entity_list.index('B-ROLE')
            role = decoded_value[B_role_index]

        return {
            'policy': '',
            'role': role,
        }