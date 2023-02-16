from tqdm import tqdm 
from underthesea import sent_tokenize, word_tokenize

import json
import numpy as np 

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

  def evaluate_dataset(self, visualize):
    print("Total number of clusters: ", len(self.clusters))
    
    array_summary_token = [] 
    array_summary_sents = []
    array_token_compression_rate = []
    array_sents_compression_rate = [] 
    array_avg_sents_per_paragraph = []
    array_avg_sents_per_document = []
    array_document = []
    array_paragraph = []
    array_sentences = []
    array_token = [] 
    
    for cluster in tqdm(self.clusters):
        
        total_token_cluster = 0
        total_sents_cluster = 0
        total_paragraphs = 0
        
        for document in cluster.documents:
            
            total_tokens_doc = 0
            total_sents_doc = 0
            
            for paragraph in document.paragraphs:
                
                total_tokens = 0
                total_sents_paragraph = len(paragraph.sentences)
                
                for sentence in paragraph.sentences:
                    total_tokens += len(sentence.split(' '))
                    
                total_tokens_doc += total_tokens 
                total_sents_doc += total_sents_paragraph
                
            total_token_cluster += total_tokens_doc
            total_sents_cluster += total_sents_doc
            total_paragraphs += len(document.paragraphs)
            
        total_token_summary = 0
        total_sents_summary = len(cluster.summary)
        
        for sent in cluster.summary:
            total_token_summary += len(sent.split(' '))
        
        array_summary_token.append(total_token_summary)
        array_summary_sents.append(total_sents_summary)
        array_token_compression_rate.append(total_token_summary / total_token_cluster)
        array_sents_compression_rate.append(total_sents_summary / total_sents_cluster)
        array_avg_sents_per_paragraph.append(total_sents_cluster / total_paragraphs)
        array_avg_sents_per_document.append(total_sents_cluster / len(cluster.documents))
        array_document.append(len(cluster.documents))
        array_paragraph.append(total_paragraphs)
        array_sentences.append(total_sents_cluster)
        array_token.append(total_token_cluster)
        
    print("Average number of token in summary: ", np.mean(array_summary_token))
    print("Average number of sents in summary: ", np.mean(array_summary_sents))
    print("Average compression rate (total token summary/total token cluster): ", np.mean(array_token_compression_rate))
    print("Average compression rate (total sents summary/total sents cluster): ", np.mean(array_sents_compression_rate))
    print("Average sentence per paragraph (total sents cluster/total paragraphs): ", np.mean(array_avg_sents_per_paragraph))
    print("Average sentence per document (total sents cluster/total document): ", np.mean(array_avg_sents_per_document))
    print("Average number of document: ", np.mean(array_document))
    print("Average number of paragraph: ", np.mean(array_paragraph))
    print("Average number of sentence: ", np.mean(array_sentences))
    print("Average number of token: ", np.mean(array_token))
    
    if visualize:
      bins = 50 
      self.visualize(array_summary_token, bins, "Number of token in summary")
      self.visualize(array_summary_sents, bins, "Number of sents in summary")
      self.visualize(array_token_compression_rate, bins, "Token compression rate")
      self.visualize(array_sents_compression_rate, bins, "Sents compression rate")
      self.visualize(array_avg_sents_per_paragraph, bins, "Average sentence per paragraph")
      self.visualize(array_avg_sents_per_document, bins, "Average sentence per document")
      self.visualize(array_document, bins, "Number of document")
      self.visualize(array_paragraph, bins, "Number of paragraph")
      self.visualize(array_sentences, bins, "Number of sentence")
      self.visualize(array_token, bins, "Number of token")

    return array_summary_token, array_summary_sents, array_token_compression_rate, array_sents_compression_rate, array_avg_sents_per_paragraph, array_avg_sents_per_document, array_document, array_paragraph, array_sentences, array_token

  def visualize(self, metrics, bins, name):
    plt.hist(metrics, bins, ec="yellow", fc="green", alpha=0.5)
    plt.title(name)
    plt.show()



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
  dataset = load_vlsp('test', False)

  dataset.evaluate_dataset()