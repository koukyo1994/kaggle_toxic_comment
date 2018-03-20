import numpy as np
import pandas as pd
import seaborn as sns
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
import lightgbm as lgbm
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import Ridge
from wordbatch.models import FM_FTRL

train_path = "../datas/train.csv"
test_path = "../datas/test.csv"

print("Read CSV.")
df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def feature_modify(df):
    comment_text = df['comment_text']
    df['Length_of_comments'] = comment_text.apply(lambda x: len(x))
    df['n_capitals'] = comment_text.apply(lambda x: len(re.findall(r"[A-Z]", x)))
    df['prop_capitals'] = df['n_capitals'] / df['Length_of_comments']
    df['n_exclamation'] = comment_text.apply(lambda x: len(re.findall(r"!", x)))
    df['n_question'] = comment_text.apply(lambda x: len(re.findall(r"\?", x)))
    df['n_punct'] = comment_text.apply(lambda x: len(re.findall(r"[#$%&*]", x)))
    df['n_words'] = comment_text.apply(lambda x: len(x.split()))
    df['n_unique'] = comment_text.apply(lambda x: len(np.unique(x.split())))
    df['prop_unique'] = df['n_unique'] / df['n_words']
    df['n_smiles'] = comment_text.apply(lambda x: len(re.findall(r":\)", x)))
    return df

new_df = feature_modify(df_train)
new_test = feature_modify(df_test)
print("Modifying the features.")

COMMENT = 'comment_text'
df_train[COMMENT].fillna("unknown", inplace=True)
df_test[COMMENT].fillna("unknown", inplace=True)
print("Fill the missing.")

re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): return re_tok.sub(r' \1 ', s).split()

print("Tfidf Vectorizing words.")
n = new_df.shape[0]
tv = TfidfVectorizer(ngram_range=(1, 3),
                     tokenizer=tokenize,
                     min_df=3,
                     max_df=0.9,
                     strip_accents='unicode',
                     use_idf=1,
                     smooth_idf=1,
                     sublinear_tf=1,
                     stop_words="english",
                     max_features=100000)
trn_term_doc = tv.fit_transform(new_df[COMMENT])
test_term_doc = tv.transform(new_test[COMMENT])

print("Concatenating the feature matrices.")
added_train = new_df.loc[:, ['n_capitals', 'prop_capitals', 'n_exclamation', 'n_unique', 'prop_unique']]
added_test = new_test.loc[:, ['n_capitals', 'prop_capitals', 'n_exclamation', 'n_unique', 'prop_unique']]
added_train = csr_matrix(added_train)
added_test = csr_matrix(added_test)

x = hstack([trn_term_doc, added_train])
test_x = hstack([test_term_doc, added_test])

print("Tfidf Vectorizing characters.")
tv = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    stop_words='english',
    ngram_range=(2, 6),
    max_features=100000)
trn_char_doc = tv.fit_transform(new_df[COMMENT])
test_char_doc = tv.transform(new_test[COMMENT])

print("Concatenating the feature matrices.")
x = hstack([trn_char_doc, x])
test_x = hstack([test_char_doc, test_x])

x = x.tocsr()
test_x = test_x.tocsr()

x, x_eval, y, y_eval = train_test_split(x,
                                        df_train.loc[:, ['toxic', 'severe_toxic', 'obscene',
                                                         'threat', 'insult', 'identity_hate']],
                                        test_size=0.1,
                                        random_state=123)

columns = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

preds_Ridge = []
for i, j in enumerate(columns):
    print('fit ', j)
    ridge = Ridge(alpha=5.0, fit_intercept=True, max_iter=100, tol=0.0025)
    ridge.fit(x, y[j])
    y_pred = ridge.predict(test_x)
    preds_Ridge.append(sigmoid(y_pred))

submission = pd.DataFrame({'id': df_test['id']})
for i, name in enumerate(columns):
    submission[name] = preds_Ridge[i]

submission.to_csv("../submission/Ridge.csv", index=False)

