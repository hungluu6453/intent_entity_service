import uvicorn
import requests
import logging
from typing import List, Dict
from datetime import datetime, timezone
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from model import *
from constant import *
from database import Database

origins = [
    "http://localhost:3000",
    "http://localhost:8002",
]
CONV_URL = "http://localhost:8003/api/v1/conversation_manage"
QA_URL = "http://localhost:8004/api/v1/retrieve"

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

data_processor = Data_Processor()
intent_classifier = Intent_Classifier(INTENT_LIST)
entity_recognizer = Entity_Recognizer(ENTITY_LIST)
database = Database()
conversation_id_list = list()

class Request_Item(BaseModel):
    conversation_id: str
    utterance: str
    voice_filename: str = None


class Response_Item(BaseModel):
    intent: str
    intent_confidence: float
    entity: List[str] = []
    entity_confidence: List[float] = []
    entity_value: List[str] = []
    response: str
    policy_response: str
    start_position: int
    end_position: int
    qa_execution_time: float
    context: str
    question: str


@app.post("/api/v1/intent_entity_classify")
def intent_entity_classify(Request: Request_Item):
    input_dt = datetime.now(timezone.utc)

    conversation_id = Request.conversation_id
    utterance = Request.utterance
    voice_filename = Request.voice_filename
    
    # Intent Classification and Named Entity Recognition
    tokenized_input = data_processor.data_processing([utterance])
    intent_response = intent_classifier.predict(tokenized_input)
    entity_response = entity_recognizer.predict(tokenized_input)
    entity_value = data_processor.tokenizer.decode(entity_response['value']).split()
    
    intent=str(intent_response['intent'])
    intent_confidence= intent_response['confidence']
    entity=entity_response['entity']
    # entity_confidence=entity_response['confidence']
    entity_value=entity_value

    entity_dict = entity_recognizer.create_entity_dict(entity, entity_value)
    # Conversation Update
    conv_response = requests.post(CONV_URL, json={'conversation_id': conversation_id, 'intent': intent, 'entity': entity_dict, 'utterance': utterance}).json()
    action = conv_response['action']
    response = conv_response['response']
    role = conv_response['role']

    start_position = -1
    end_position = -1
    qa_execution_time = -1
    context = ""
    policy_response = ""
    paragraph_id = None

    # Question Answering
    if action == 'answer':
        qa_response = requests.post(QA_URL, json={'role': role, 'question': conv_response['question']}).json()
        policy_response = qa_response['text']
        if qa_response['text'] != "":
            response = response
        else:
            response = "Xin lỗi nha, mình không tìm được câu trả lời cụ thể. Tuy nhiên, đoạn quy định sau có thể liên quan đến điều bạn thắc mắc."

        start_position = qa_response['start_position']
        end_position = qa_response['end_position']
        qa_execution_time = qa_response['execution_time']
        context = qa_response['context']
        paragraph_id = qa_response['paragraph_id']

    output_dt = datetime.now(timezone.utc)
    voice_id = None
    # Insert voice into the database
    if voice_filename:
        database.insert_voice(voice_filename)
        voice_id = database.get_voiceid()
    # Insert conversation into the database
    if conversation_id not in conversation_id_list:
        conversation_id_list.append(conversation_id)
        database.insert_conversation(conversation_id)
    # Insert input and output into the database
    database.insert_utterance(voice_id, utterance, True, conversation_id, input_dt, None)
    database.insert_utterance(None, response + policy_response, False, conversation_id, output_dt, paragraph_id)

    logging.info('Conversation_id: %s, Utterance: %s', conversation_id, utterance)
    logging.info('Inent: %s', intent)
    logging.info('Entity: %s', entity_dict)
    logging.info('Response: %s', response)
    logging.info('Answer: %s', policy_response)

    return Response_Item(
        intent=intent,
        intent_confidence=intent_confidence,
        entity=entity,
        entity_confidence=[1.],
        entity_value=entity_value,
        response=response,
        policy_response=policy_response,
        start_position=start_position,
        end_position=end_position,
        qa_execution_time=qa_execution_time,
        context=context,
        question=utterance,
    )

if __name__ == '__main__':
    uvicorn.run("app:app", host='0.0.0.0', port=8001)
    database.close_connection()