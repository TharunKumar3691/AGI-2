import numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

tr = pd.read_csv("train.csv"); te = pd.read_csv("test.csv")
def fe(df):
    d = df.copy()
    d["Title"] = d.Name.str.extract(r",\s*([^\.]+)\.")[0].str.strip()
    d["Title"] = d.Title.replace({"Mlle":"Miss","Ms":"Miss","Mme":"Mrs","Lady":"Rare","Countess":"Rare","Capt":"Rare","Col":"Rare","Don":"Rare","Dr":"Rare","Major":"Rare","Rev":"Rare","Sir":"Rare","Jonkheer":"Rare","Dona":"Rare"})
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
all_ = pd.concat([tr.drop(columns="Survived"), te], keys=["tr","te"])
X = fe(all_); Xtr, Xte = X.loc["tr"], X.loc["te"]; y = tr.Survived
cv = StratifiedKFold(10, shuffle=True, random_state=42)
models = {
 "logreg": make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=2000)),
 "rf": RandomForestClassifier(500, min_samples_leaf=3, max_features="sqrt", random_state=42, n_jobs=-1),
 "gb": GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=42),
 "hgb": HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=150, l2_regularization=1.0, random_state=42),
}
for n, m in models.items():
    s = cross_val_score(m, Xtr, y, cv=cv); print(f"{n:7s} {s.mean():.4f} ±{s.std():.4f}")
ens = VotingClassifier(list(models.items()), voting="soft")
s = cross_val_score(ens, Xtr, y, cv=cv); print(f"{'ens':7s} {s.mean():.4f} ±{s.std():.4f}")
