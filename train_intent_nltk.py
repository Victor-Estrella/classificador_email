import os
import csv
import re
import pickle
from typing import List, Tuple

import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

DATA_PATH = os.path.join('data', 'training', 'emails.csv')
MODEL_PATH = os.path.join('models', 'intent_nb.pkl')


def ensure_nltk():
    try:
        stopwords.words('portuguese')
    except LookupError:
        nltk.download('stopwords')


TOKEN_RE = re.compile(r"[\wÀ-ÿ]+", re.UNICODE)


def tokenize(text: str) -> List[str]:
    text = text.lower()
    return TOKEN_RE.findall(text)


def normalize(tokens: List[str]) -> List[str]:
    stemmer = SnowballStemmer('portuguese')
    sw = set(stopwords.words('portuguese'))
    norm = []
    for t in tokens:
        # ignora números puros e ids
        if t.isdigit() or re.fullmatch(r"[0-9\-#]+", t):
            continue
        if t in sw:
            continue
        st = stemmer.stem(t)
        # ignora stems curtíssimos que pouco informam
        if len(st) < 3:
            continue
        norm.append(st)
    return norm


def featurize(tokens: List[str]):
    return {f"has({t})": True for t in tokens}


def load_dataset(path: str) -> List[Tuple[dict, str]]:
    data = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row['text']
            label = row['label']
            feats = featurize(normalize(tokenize(text)))
            data.append((feats, label))
    return data


def train():
    ensure_nltk()
    dataset = load_dataset(DATA_PATH)
    if not dataset:
        raise SystemExit('Dataset vazio. Adicione exemplos em data/training/emails.csv')
    # Split simples: 80/20
    split = int(0.8 * len(dataset))
    train_set = dataset[:split]
    test_set = dataset[split:] if split < len(dataset) else dataset

    clf = nltk.NaiveBayesClassifier.train(train_set)

    # Avaliação
    accuracy = nltk.classify.accuracy(clf, test_set)
    print(f"Acurácia de validação: {accuracy:.2%} ({len(test_set)} exemplos)")
    try:
        clf.show_most_informative_features(10)
    except Exception:
        pass

    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(clf, f)
    print(f"Modelo salvo em {MODEL_PATH}")


if __name__ == '__main__':
    train()
