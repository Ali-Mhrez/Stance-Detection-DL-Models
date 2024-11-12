import json
import numpy as np
from sklearn.metrics import f1_score
import nltk
from nltk.corpus import stopwords

stance_to_int = {'Agree': 0, 'Disagree': 1, 'Discuss': 2, 'Unrelated': 3, \
                 'agree': 0, 'disagree': 1, 'discuss': 2, 'unrelated': 3, \
                 'other': 2} # some datasets merge discuss and unrelated

class AraStanceData:
  """
  Class for AraStance dataset
  """

  def __init__(self, path):
    """
    Args:
      path: str, path/filename.extension
    """
    self.claims, self.articles, self.stances, self.article_claim = self.read(path)
    
  def read(self, path):
    """
    Read Arastance data from jsonl files
    Args:
      path: str, path/filename.extension
    Returns:
      claims: list, all claims
      articles: list, all articles
      stances: list, articles stances
      article_claim: list, mapping between articles and claims
    """
    claims, articles, stances = [], [], []
    article_claim = []
    with open(path, 'r', encoding='utf-8') as file:
      for idx, line in enumerate(file):
        instance = json.loads(line)
        claims.append(instance['claim'])
        articles.extend(instance['article'])
        article_claim.extend([idx] * len(instance['article']))
        stances.extend(instance['stance'])
    return claims, articles, stances, article_claim
  
def load_stop_words():
  """
  Load stop words from the file path.
  Args:
  Returns:
    nltk_stop_words: list, nltk stop words list
    custom_stop_words: list, list from the file path
  """
  nltk.download('stopwords')
  nltk_stop_words = stopwords.words('arabic')
  
  custom_stop_words = [
  'هى', 'وهى', 'انه', 'خلال', 'كانت', 'وفي', 'في', 'التي', 'الذي', 'ان',
  'الى', 'او', 'اي', 'انها', 'الا', 'اما', 'وان', 'فى', 'تم', 'ويتم', 'أنه', 
  'آمين', 'أب', 'أخ', 'أفعل', 'أفعله', 'ؤلاء', 'إل', 'إم', 'ات', 'اتان', 
  'ارتد', 'انفك', 'برح', 'تان', 'تبد', 'تحو', 'تعل', 'حد', 'حم', 'حي', 
  'خب', 'ذار', 'سيما', 'صه', 'ظل', 'ظن', 'عد', 'قط', 'مر', 'مكان', 
  'مكانكن', 'نب', 'هات', 'هب', 'واها', 'وراء'
  ]
  
  return nltk_stop_words, custom_stop_words
  
def evaluate(model, dataset, y_true):

  loss, accuracy = model.evaluate(dataset)
  predictions = model.predict(dataset)
  
  if predictions.shape[1] == 4:
    y_pred = np.argmax(predictions, axis=1)
  else:
    y_pred = np.argmax(predictions[0], axis=1)
  
  accuracy = len([i for i in range(len(y_pred)) if y_pred[i] == y_true[i]]) / len(y_true)
  f1score = f1_score(y_true, y_pred, average=None)
  mf1score = f1_score(y_true, y_pred, average='macro')

  return loss, accuracy, f1score, mf1score