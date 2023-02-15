from tqdm.notebook import tqdm 

class Dataset:
  def __init__(self, is_test: bool):
    self.is_test = is_test 
    self.clusters = []
  
class Cluster:
  def __init__(self):
    pass 

  def add_document(self):
    pass 

  def get_all_sents(self):
    pass 

class Document:
  def __init__(self, raw: str):
    pass

def load_vlsp(path, is_test, partial):
  dataset = Dataset()

  with open(path, 'r') as json_file:
      json_list = list(json_file)

      print(json_list[0])

if __name__ == '__main__':
  load_vlsp(
    '/home/dang/Summarization-dataset-reader/dataset/vlsp/vlsp_2022_abmusu_train_data_new.jsonl', 
    False, True):