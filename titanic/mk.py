import nbformat as nbf
nb = nbf.v4.new_notebook()
md = lambda s: nbf.v4.new_markdown_cell(s); code = lambda s: nbf.v4.new_code_cell(s)
nb.cells = [
md("# Titanic: feature engineering + soft-voting ensemble\nLogReg + RandomForest + GradientBoosting + HistGradientBoosting, 10-fold stratified CV, fixed seeds. Writes `submission.csv`."),
code('''import os, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import glob
hits = glob.glob("/kaggle/input/**/train.csv", recursive=True)
DATA = os.path.dirname(hits[0]) if hits else "."
print("data dir:", DATA)
tr = pd.read_csv(f"{DATA}/train.csv"); te = pd.read_csv(f"{DATA}/test.csv")
print(tr.shape, te.shape)'''),
code('''RARE = ["Lady","Countess","Capt","Col","Don","Dr","Major","Rev","Sir","Jonkheer","Dona"]
def fe(df):
    d = df.copy()
    d["Title"] = d.Name.str.extract(r",\\s*([^\\.]+)\\.")[0].str.strip().replace({"Mlle":"Miss","Ms":"Miss","Mme":"Mrs", **{t:"Rare" for t in RARE}})
    d["FamilySize"] = d.SibSp + d.Parch + 1
    d["IsAlone"] = (d.FamilySize == 1).astype(int)
    d["Deck"] = d.Cabin.str[0].fillna("U")
    d["HasCabin"] = d.Cabin.notna().astype(int)
    d["TicketCount"] = d.groupby("Ticket").Ticket.transform("count")
    d["Fare"] = d.Fare.fillna(d.Fare.median()); d["LogFare"] = np.log1p(d.Fare)
    d["FarePP"] = d.Fare / d.TicketCount
    d["Embarked"] = d.Embarked.fillna("S")
    d["Age"] = d.Age.fillna(d.groupby(["Title","Pclass"]).Age.transform("median")).fillna(d.Age.median())
    d["Child"] = (d.Age < 13).astype(int)
    d["Sex"] = (d.Sex == "male").astype(int)
    cols = ["Pclass","Sex","Age","SibSp","Parch","LogFare","FarePP","FamilySize","IsAlone","HasCabin","TicketCount","Child","Title","Deck","Embarked"]
    return pd.get_dummies(d[cols], columns=["Title","Deck","Embarked"], dtype=int)

X = fe(pd.concat([tr.drop(columns="Survived"), te], keys=["tr","te"]))
Xtr, Xte = X.loc["tr"], X.loc["te"]; y = tr.Survived
assert list(Xtr.columns) == list(Xte.columns) and not Xtr.isna().any().any() and not Xte.isna().any().any()
Xtr.shape'''),
code('''models = {
 "logreg": make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000)),
 "rf": RandomForestClassifier(500, min_samples_leaf=3, max_features="sqrt", random_state=42, n_jobs=-1),
 "gb": GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=42),
 "hgb": HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, l2_regularization=1.0, random_state=42),
}
cv = StratifiedKFold(10, shuffle=True, random_state=42)
for n, m in models.items():
    s = cross_val_score(m, Xtr, y, cv=cv); print(f"{n:7s} CV acc {s.mean():.4f} +/- {s.std():.4f}")
ens = VotingClassifier(list(models.items()), voting="soft")
s = cross_val_score(ens, Xtr, y, cv=cv); print(f"ensemble CV acc {s.mean():.4f} +/- {s.std():.4f}")'''),
code('''ens.fit(Xtr, y)
sub = pd.DataFrame({"PassengerId": te.PassengerId, "Survived": ens.predict(Xte).astype(int)})
sub.to_csv("submission.csv", index=False)
assert len(sub) == 418 and set(sub.Survived) <= {0, 1} and sub.PassengerId.is_unique
print(sub.Survived.mean(), sub.shape); sub.head()'''),
]
nb.metadata = {"kernelspec":{"name":"python3","display_name":"Python 3","language":"python"},"language_info":{"name":"python"}}
nbf.write(nb, "nb/titanic-ensemble.ipynb")
