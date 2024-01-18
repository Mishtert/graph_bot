from pymongo.mongo_client import MongoClient

from urllib.parse import quote_plus

username = quote_plus("davyjdemo")
password = quote_plus("chatdb123!")
uri = f"mongodb+srv://{username}:{password}@cluster0.ddlyqbd.mongodb.net/?retryWrites=true&w=majority"
client = MongoClient(uri)

# Get Database
db = client["it_support"]

# Get Collection
collection = db['it_support_ticket_data']
# print(client["it_support"]["it_support_ticket_data"])

def insert_data(table=collection, data=None):
    table.insert_many(data)


import json

# Specify the path to your JSON file
json_file_path = "./data/raw_data/db_data/it_support.json"

# Read the JSON file
with open(json_file_path, "r") as file:
    ticket_data = json.load(file)

insert_data(table=collection, data=ticket_data)