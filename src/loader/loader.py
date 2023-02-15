from tqdm import tqdm 
from underthesea import sent_tokenize, word_tokenize

import json

class Dataset:
  def __init__(self, is_test: bool):
    self.is_test = is_test 
    self.clusters = []

  def add_cluster(self, cluster):
    self.clusters.append(cluster)

  def show(self):
    for i, cluster in enumerate(self.clusters):
      print("Cluster {}".format(i))
      cluster.show()


class Cluster:
  def __init__(self, summary, category):
    self.category = category 
    self.summary = summary 
    self.documents = []

  def add_document(self, doc):
    self.documents.append(doc)

  def get_all_sents(self):
    return [document.get_all_sents() for document in self.documents]

  def show(self):
    print("Category: ", self.category)
    print("Summary: ", self.summary)
    for i, document in enumerate(self.documents):
      print("Document {}".format(i))
      document.show()


class Document:
  def __init__(self, title, anchor_text):
    self.title = title 
    self.anchor_text = anchor_text 
    self.paragraphs = []

  def add_paragraph(self, paragraph):
    self.paragraphs.append(paragraph)

  def get_all_sents(self):
    return [paragraph.get_all_sents() for paragraph in self.paragraphs]

  def show(self):
    print("Title: ", self.title)
    print("Anchor text: ", self.anchor_text)
    for paragraph in self.paragraphs:
      print('[')
      paragraph.show()
      print(']')


class Paragraph:
  def __init__(self, sentences):
    self.sentences = sentences 

  def get_all_sents(self):
    return self.sentences

  def show(self):
    print(' '.join(self.sentences))


def tokenize(paragraph):
  sents = sent_tokenize(paragraph)
  return [word_tokenize(sent, format="text") for sent in sents]

def load_vlsp(type, partial):
  if type == 'train':
    path = './dataset/vlsp/vlsp_2022_abmusu_train_data_new.jsonl'
    is_test = False
  elif type == 'valid':
    path = './dataset/vlsp/vlsp_2022_abmusu_validation_data_new.jsonl'
    is_test = False
  elif type == 'test':
    path = './dataset/vlsp/vlsp_abmusu_test_data.jsonl'
    is_test = True

  dataset = Dataset(is_test)

  with open(path, 'r') as json_file:
    json_list = list(json_file)

    json_dto = json.loads(json_list[0])

    print(json_dto.keys())
    print(json_dto['single_documents'][0].keys())

    for idx, cluster in tqdm(enumerate(json_list)):
      if partial:
        if idx > 1:
          break 

      json_dto = json.loads(cluster)

      if not is_test:
        summary = tokenize(json_dto['summary'])
      else:
        summary = []
        
      new_cluster = Cluster(summary, json_dto['category'])

      for document in json_dto['single_documents']:
        title = tokenize(document['title'])
        anchor_text = tokenize(document['anchor_text'])

        new_document = Document(title, anchor_text)

        raw_text = document['raw_text']
        for paragraph in raw_text.split('\n'):
          sents = tokenize(paragraph)

          paragraph = Paragraph(sents)
          new_document.add_paragraph(paragraph)

        new_cluster.add_document(new_document)

      dataset.add_cluster(new_cluster)

  return dataset 

if __name__ == '__main__':
  dataset = load_vlsp('test', True)

  dataset.show()