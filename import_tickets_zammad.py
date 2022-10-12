#!/usr/bin/env python3

import requests
import pandas as pd
import httpimport
import ticket

r = requests.get('https://datasets-server.huggingface.co/first-rows?dataset=milesbutler%2Fconsumer_complaints&config=milesbutler--consumer_complaints&split=train')
json_data=r.json()['rows']
    
for ticket_idx in range(50):
    ticket_obj=ticket.TicketAPI()
    
    title='Complaint'+' '+str(ticket_idx)
    subject=json_data[0]['row']['Issue']+' '+str(ticket_idx)
    body_text=json_data[0]['row']['Consumer Complaint']

    ticket_obj.create_ticket_with_article(title, subject, body_text)
   
