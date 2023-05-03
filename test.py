from model import *
from constant import *
from vncorenlp import VnCoreNLP


# annotator = VnCoreNLP('resource/vncorenlp/VnCoreNLP-1.2.jar', annotators='wseg')
# def annotate(x):
#     return ' '.join([' '.join(text) for text in annotator.tokenize(x)])
# print(annotate('Câu trả lời của bạn đây'))

data_processor = Data_Processor()
intent_classifier = Intent_Classifier(INTENT_LIST)
entity_recognizer = Entity_Recognizer(ENTITY_LIST)

utterance = 'cho mình hỏi về việc đóng học phí của thạc sĩ'

tokenized_input = data_processor.data_processing([utterance])
intent_response = intent_classifier.predict(tokenized_input)
entity_response = entity_recognizer.predict(tokenized_input)
value = [data_processor.tokenizer.decode(entity_response['value'])]

print((
        intent_response['intent'],
        intent_response['confidence'],
        entity_response['entity'],
        entity_response['confidence'],
        value,
    ))
