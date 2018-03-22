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
import gc

train_path = "../datas/train.csv"
test_path = "../datas/test.csv"

print("Read CSV.")
df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

COMMENT = 'comment_text'
df_train[COMMENT].fillna("unknown", inplace=True)
df_test[COMMENT].fillna("unknown", inplace=True)
print("Fill the missing.")

re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): return re_tok.sub(r' \1 ', s).split()

print("Tfidf Vectorizing words.")
n = df_train.shape[0]
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
trn_term_doc = tv.fit_transform(df_train[COMMENT])
test_term_doc = tv.transform(df_test[COMMENT])

print("Tfidf Vectorizing characters.")
tv = TfidfVectorizer(
    sublinear_tf=True,
    strip_accents='unicode',
    analyzer='char',
    stop_words='english',
    ngram_range=(2, 6),
    max_features=100000)
trn_char_doc = tv.fit_transform(df_train[COMMENT])
test_char_doc = tv.transform(df_test[COMMENT])

del tv
gc.collect()

print("Concatenating the feature matrices.")
x = hstack([trn_char_doc, trn_term_doc])
test_x = hstack([test_char_doc, test_term_doc])

x = x.tocsr()
test_x = test_x.tocsr()

print("Splitting the dataset.")
x, x_eval, y, y_eval = train_test_split(x,
                                        df_train.loc[:, ['toxic', 'severe_toxic', 'obscene',
                                                         'threat', 'insult', 'identity_hate']],
                                        test_size=0.1,
                                        random_state=123)

columns = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

preds_FMFTRL = []
for i, j in enumerate(columns):
    print('fit ', j)
    fm_ftrl = FM_FTRL(
                    alpha=0.02, beta=0.01, L1=0.00001, L2=30.0,
                    D=x.shape[1], alpha_fm=0.1,
                    L2_fm=0.5, init_fm=0.01, weight_fm= 50.0,
                    D_fm=200, e_noise=0.0, iters=3,
                    inv_link="identity", threads=8
                )
    fm_ftrl.fit(x, y[j])
    y_pred = fm_ftrl.predict(test_x)
    preds_FMFTRL.append(sigmoid(y_pred))

submission = pd.DataFrame({'id': df_test['id']})
for i, name in enumerate(columns):
    submission[name] = preds_FMFTRL[i]

submission.to_csv("../submission/FMFTRL.csv", index=False)
