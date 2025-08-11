import re
import pyarabic
import pyarabic.trans
import pyarabic.number

def remove_puncuation(text):
  text = re.sub("،", " ", text)
  text = re.sub("\.", " ", text)
  text = re.sub("[«»”“؛]", " ", text)
  text = re.sub("\?", " \? ", text)
  text = re.sub("؟", " ؟ ", text)
  text = re.sub("\"", " ", text)
  text = re.sub("[():,'!;+*_|]", " ", text)
  text = re.sub("\\\\", " ", text)
  text = re.sub("/", " ", text)
  text = re.sub("\[.+\]", " ", text)
  # - could be part of a date, soccer result, and compound word
  text = re.sub("-", " ", text)
  return text

def remove_diacritics(text):
  arabic_diacritics = re.compile("""
                             ّ    | # Tashdid
                             َ    | # Fatha
                             ً    | # Tanwin Fath
                             ُ    | # Damma
                             ٌ    | # Tanwin Damm
                             ِ    | # Kasra
                             ٍ    | # Tanwin Kasr
                             ْ    | # Sukun
                             ـ     # Tatwil/Kashida
                         """, re.VERBOSE)
  text = re.sub(arabic_diacritics, '', text)
  return text

def remove_longation(text):
  p_longation = re.compile(r'(.)\1+')
  subst = r"\1\1"
  text = re.sub(p_longation, subst, text)
  return text

def remove_unicode_codes(text):
  text = re.sub("\\u202c", " ", text)
  text = re.sub("\\u202b", " ", text)
  text = re.sub("\\u200f", " ", text)
  return text

def normalize(text):
  text = re.sub("[إأآ]", "ا", text)
  text = re.sub("ى", "ي", text)
  text = re.sub("ة", "ه", text)
  text = re.sub("اا", "ا", text)
  return text

def process_numerals(text):
  # convert all numerals to western arabic
  text = pyarabic.trans.normalize_digits(text, source='all', out='west')
  # add a space between numerals and their surroundings
  text = re.sub(r"([0-9]+(\.[0-9]+)?(\.[0-9]+)?)",r" \1 ", text)
  text = re.sub("٪", "%", text)
  text = re.sub("%",r" بالمئه ", text)

  def int2str(match):
    an = pyarabic.number.ArNumbers()
    group = match.group(1)
    if len(group) > 10:
      return group
    return an.int2str(group)
  text = re.sub('([0-9]+)', int2str, text)

  return text

# def remove_shorts(text):
#   text = re.sub(r"\b[ا-ي]{,2}\b", " ", text)
#   return text

def remove_extra_spaces(text):
  text = re.sub(' +', ' ', text)
  return text