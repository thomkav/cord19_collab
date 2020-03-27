from pymongo import MongoClient, TEXT

TEXT_QUERY = [
    {"$addFields": 
        {"refs":{"$objectToArray":"$ref_entries"},
         "bibs": {"$objectToArray": "$bib_entries"}
        }
    },
    {"$project": 
         {"all_text":
              {"$concatArrays":
                   [
                    ["$metadata.title"],
                    "$abstract.text",
                    "$body_text.text",
                   "$refs.v.text",
                   "$bibs.v.title",]
              },
         }
    },
    # code below produces full text as one string
#     {"$addFields":
#          {"full_text": 
#               {"$reduce":
#                    {"input":
#                            {"$concatArrays":
#                                 ["$body_and_abstract", "$refs.v.text"]},
#                        "initialValue": "",
#                        "in": {"$concat":["$$value","$$this"]}
#                    }
#               }
#          }
     
#     },
    {"$unwind": "$all_text"}
]
CONTEXT_QUERY = [
    {"$addFields": 
        {"refs": {"$objectToArray": "$ref_entries"},
        "bibs": {"$objectToArray": "$bib_entries"}
        }
#         "metadata":1,
#         "abstract":1,
#         "body_text":1}
    },
     {"$project": 
          {"context":
               {"$concatArrays":
                    [
                    ["$metadata"],
                     "$abstract","$body_text", 
                    "$bibs","$refs"]
               },
           "metadata": 1,
           "paper_id": 1,
          }
     },
     {"$unwind": "$context"},
     {"$project": {"context.v.title":0,
                  "context.v.text":0,
                  "context.text":0,
                  "context.title":0}
     },
]

def make_doc_gen(db, query=None):
    """
    Returns tuples of of the form (text, context) from the mongo database. 
    
    Each text instance is one paragraph from the embedded documents in a paper, and the context is everything in that embedded document that is not text (i.e., cite spans, ref spans, section info, location info) as well as the metadata and paper_id for the overall document.
    
    These tuples can then be passed to a spacy pipeline and use the context for custom extensions.
    
    :Parameters:
        - `db`: a pymongo database object
        - `query`: a dictionary-style query as used by pymongo. E.g., {"metadata.title": {"$regex": "coronavirus"}}
        .. if the query uses the $text operator, the returned documents are sorted by textScore.
    """
    
    con = CONTEXT_QUERY
    text = TEXT_QUERY
    # add match stage if we want to filter documents with a query before aggregation.
    if query is not None:
        con = [{"$match": query}] + CONTEXT_QUERY
        text = [{"$match": query}] + TEXT_QUERY
        # if we use a text query, need to sort results
        if query.get('$text', None) is not None:
            con.append({"$sort": {"score":
                                  {"$meta": "textScore"},
                                 '_id':1}
                       })
            text.append({"$sort": {"score":
                                  {"$meta": "textScore"},
                                  "_id":1}
                       })
    for collection in db.list_collection_names():
        context = db[collection].aggregate(con, allowDiskUse=True)
        # spacy wants the first element of the tuple passed to nlp.pipe to be a string, 
        # so we pull the text out of the dictionary
        full_text = (doc['all_text'] for doc in db[collection].aggregate(text, allowDiskUse=True))
        
        yield from zip(full_text, context)

        
def search_collections(query, proj=None):
    """
    Generator that returns JSON documents if they match search terms in the abstract. Searches across collections for easy 
    matching.
    
    :Parameters:
        - `query`: a dictionary-style query as used by pymongo. E.g., {"metadata.title": {"$regex": "coronavirus"}}
        ` `proj`: optional projection for returned documents. Use pymongo style projections, e.g., {"metadata": 1}
    """
    collections = [db[collection] for collection in db.list_collection_names()]
    for collection in collections:
        yield from collection.find(query, proj)
        
