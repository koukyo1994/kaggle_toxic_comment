import pandas as pd

lgbm = pd.read_csv("../submission/LGBM.csv")
ridge = pd.read_csv("../submission/Ridge.csv")
fmftrl = pd.read_csv("../submission/FMFTRL.csv")

columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
lgbm_val = lgbm.loc[:, columns].values
ridge_val = ridge.loc[:, columns].values
fmftrl_val = fmftrl.loc[:, columns].values

total_val = 0.2*lgbm_val + 0.4*ridge_val + 0.4*fmftrl_val

submission = pd.DataFrame()
submission['id'] = lgbm['id']
for i,name in enumerate(columns):
    submission[name] = total_val[:, i]

submission.to_csv("../submission/Ensemble.csv", index=False)
