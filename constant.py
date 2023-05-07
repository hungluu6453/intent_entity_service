INTENT_LIST = [
    'policy',
    'answer',
    'greeting',
    'confirm',
    'disagree',
    'chitchat',
    'faq',
    # 'find',
]
ENTITY_LIST = [
    'policy',
    'role',
    # 'department',
    # 'lecturer',
    # 'major',
    #' subject',
    # 'year',
]
VNCORENLP_PATH = 'resource/vncorenlp'
TOKENIZER_PATH = 'resource/tokenizer'
INTENT_MODEL_PATH = 'resource/intent_classifier_case'
ENTITY_MODEL_PATH = 'resource/named_entity_recognizer_case'

INTENT_THRESHOLD = 0.5
ENTITY_THRESHOLD = 0.5