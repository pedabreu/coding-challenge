#!/usr/bin/env python3

import requests
import pandas as pd
import httpimport
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import ticket
    
tokenizer = AutoTokenizer.from_pretrained("finiteautomata/bertweet-base-sentiment-analysis")
model = AutoModelForSequenceClassification.from_pretrained("finiteautomata/bertweet-base-sentiment-analysis")

all_ticket_list=ticket.TicketAPI().list_tickets()

for ticket_info in all_ticket_list:  
    ticket_obj= ticket.TicketAPI()
    cur_ticket=ticket_obj.get_articles_by_ticket_id(ticket_info['id'])
    inputs = tokenizer(cur_ticket[0]['body'], return_tensors="pt")

    with torch.no_grad():
        res=model(**inputs)     
        logits = model(**inputs).logits

    predicted_class_id = logits.argmax().item()
    pred_class_id = model.config.id2label[predicted_class_id]

    switcher = {
        'POS': 1,
        'NEU': 2,
        'NEG': 3,
    }

    priority_id =switcher.get(pred_class_id, 1)   
    ticket_obj.update_ticket(ticket_info['id'], priority_id)
