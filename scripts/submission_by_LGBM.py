import numpy as np
import pandas as pd
import seaborn as sns
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix
import lightgbm as lgbm
from sklearn.cross_validation import train_test_split

train_path = "../datas/train.csv"
test_path = "../datas/test.csv"

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)

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
COMMENT = 'comment_text'
df_train[COMMENT].fillna("unknown", inplace=True)
df_test[COMMENT].fillna("unknown", inplace=True)

re_tok = re.compile(f'([{string.punctuation}“”¨«»®´·º½¾¿¡§£₤‘’])')
def tokenize(s): return re_tok.sub(r' \1 ', s).split()

n = new_df.shape[0]
tv = TfidfVectorizer(ngram_range=(1, 4), tokenizer=tokenize,
               min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
               smooth_idf=1, sublinear_tf=1 )
trn_term_doc = tv.fit_transform(new_df[COMMENT])
test_term_doc = tv.transform(new_test[COMMENT])

added_train = new_df.loc[:, ['n_capitals', 'prop_capitals', 'n_exclamation', 'n_unique', 'prop_unique']]
added_test = new_test.loc[:, ['n_capitals', 'prop_capitals', 'n_exclamation', 'n_unique', 'prop_unique']]
added_train = csr_matrix(added_train)
added_test = csr_matrix(added_test)

x = hstack([trn_term_doc, added_train])
test_x = hstack([test_term_doc, added_test])

x = x.tocsr()
test_x = test_x.tocsr()

x, x_eval, y, y_eval = train_test_split(x,
                                        df_train.loc[:, ['toxic', 'severe_toxic', 'obscene',
                                                         'threat', 'insult', 'identity_hate']],
                                        test_size=0.1,
                                        random_state=123)

X_toxic = lgbm.Dataset(x, y['toxic'])
X_severe = lgbm.Dataset(x, y['severe_toxic'])
X_obscene = lgbm.Dataset(x, y['obscene'])
X_threat = lgbm.Dataset(x, y['threat'])
X_insult = lgbm.Dataset(x, y['insult'])
X_hate = lgbm.Dataset(x, y['identity_hate'])

X_toxic_eval = lgbm.Dataset(x_eval, y_eval['toxic'])
X_severe_eval = lgbm.Dataset(x_eval, y_eval['severe_toxic'])
X_obscene_eval = lgbm.Dataset(x_eval, y_eval['obscene'])
X_threat_eval = lgbm.Dataset(x_eval, y_eval['threat'])
X_insult_eval = lgbm.Dataset(x_eval, y_eval['insult'])
X_hate_eval = lgbm.Dataset(x_eval, y_eval['identity_hate'])

X_list = [X_toxic, X_severe, X_obscene, X_threat, X_insult, X_hate]
X_eval = [X_toxic_eval, X_severe_eval, X_obscene_eval, X_threat_eval, X_insult_eval, X_hate_eval]

params = {
        'task': 'train',
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': {'auc'},
        'num_class': 1,
        'learning_rate': 0.1,
        'num_leaves': 31,
        'min_data_in_leaf': 20,
        'num_threads': 8,
        'verbose': 0
}

preds = []
for i, j in enumerate(columns):
    print('fit ', j)
    gbm = lgbm.train(params,
                     X_list[i],
                     valid_sets=X_eval[i],
                     num_boost_round=2000,
                     early_stopping_rounds=100)
    y_pred = gbm.predict(test_x, num_iteration=gbm.best_iteration)
    preds.append(y_pred)

submission = pd.DataFrame({'id': df_test['id']})
for i, name in enumerate(columns):
    submission[name] = preds[i]

submission.to_csv("../submission/LGBM.csv", index=False)
