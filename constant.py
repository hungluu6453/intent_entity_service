INTENT_LIST = [
    'policy',
    'answer',
    'greeting',
    'confirm',
    'disagree',
    'chitchat',
    'thanks',
]

ENTITY_LIST = [
    'policy',
    'role',
]
# VNCORENLP_PATH = 'resource/vncorenlp'
VNCORENLP_PATH = '7_model_repository/0_vncorenlp'
VNCORENLP_MODEL_PATH = '7_model_repository/0_vncorenlp/VnCoreNLP-1.2.jar'

TOKENIZER_PATH = 'resource/tokenizer'
# INTENT_MODEL_PATH = 'resource/intent_model_10062023'
INTENT_MODEL_PATH = '7_model_repository/1_intent_classifier'

# ENTITY_MODEL_PATH = 'resource/named_entity_recognizer_case'
ENTITY_MODEL_PATH = '7_model_repository/2_named_entity_recognizer'

INTENT_THRESHOLD = 0.5
ENTITY_THRESHOLD = 0.5