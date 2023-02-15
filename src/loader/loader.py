from tqdm.notebook import tqdm 

import json

class Dataset:
  def __init__(self, is_test: bool):
    self.is_test = is_test 
    self.clusters = []

  def add_cluster(cluster):
    self.clusters.append(cluster)


class Cluster:
  def __init__(self, summary, category):
    self.category = category 
    self.summary = summary 
    self.documents = []

  def add_document(self, doc):
    self.documents.append(doc)

  def get_all_sents(self):
    sents = []
    for doc in self.documents:
      sents = sents + doc.get_all_sents()

    return sents 


class Document:
  def __init__(self, sentences):
    self.sentences = sentences

  def get_all_sents(self):
    return self.sentences 


def load_vlsp(path, is_test, partial):
  dataset = Dataset(is_test)

  with open(path, 'r') as json_file:
    json_list = list(json_file)

    json_dto = json.loads(json_list[0])

    print(json_dto.keys())
    print(json_dto['single_documents'][0].keys())

    for cluster in json_list[0]:
      json_dto = json.load(cluster)

      cluster = Cluster()

      for document in json_dto['single_documents']:
        pass


if __name__ == '__main__':
  load_vlsp(
    './dataset/vlsp/vlsp_2022_abmusu_train_data_new.jsonl', 
    False, True)