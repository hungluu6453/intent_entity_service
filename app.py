import uvicorn
import requests
import logging
from typing import List, Dict
from datetime import datetime, timezone
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

from model import *
from constant import *


logging.basicConfig(level=logging.INFO)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

data_processor = Data_Processor()
intent_classifier = Intent_Classifier(INTENT_LIST)
entity_recognizer = Entity_Recognizer(ENTITY_LIST)


class Request_Item(BaseModel):
    conversation_id: str = None
    utterance: str = None

class Response_Item(BaseModel):
    intent: str = None
    intent_confidence: float = None
    entity_dict: Dict[str, List]


@app.post("/bkheart/api/intent_entity_classify")
def intent_entity_classify(Request: Request_Item):

    conversation_id = Request.conversation_id
    utterance = Request.utterance

    if not conversation_id:
        raise HTTPException(status_code=422, detail="Missing Conversation ID")

    if not utterance:
        raise HTTPException(status_code=422, detail="Missing Utterance")
    
    try: 
        # Intent Classification and Named Entity Recognition
        tokenized_input = data_processor.data_processing([utterance])
        intent_response = intent_classifier.predict(tokenized_input)
        entity_response = entity_recognizer.predict(tokenized_input)

        intent=intent_response['intent']
        intent_confidence= intent_response['confidence']
        bio_entity=entity_response['bio_entity']
        entity_confidence=entity_response['confidence']
        entity_value = data_processor.tokenizer.decode(entity_response['value']).split()
        
        entity_dict = entity_recognizer.create_entity_dict(bio_entity, entity_value, entity_confidence)

        logging.info('Inent: %s', intent)
        logging.info('Intent_confidence: %s', intent_confidence)
        logging.info('BIO_Entity: %s', bio_entity)
        logging.info('Entity_confidence: %s', entity_confidence)
        logging.info('Entity_value: %s', entity_value)
        logging.info('Entity_dict: %s', entity_dict)
    except:
        logging.exception("Error in the processing")
        raise HTTPException(status_code=500, detail="Intent Entity Classification Server Error")
        
    return Response_Item(
        intent=intent,
        intent_confidence=intent_confidence,
        entity_dict=entity_dict,
    )


if __name__ == '__main__':
    uvicorn.run("app:app", host='0.0.0.0', port=8001)
    