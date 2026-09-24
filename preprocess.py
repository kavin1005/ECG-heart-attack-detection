import os, numpy as np, pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import shuffle
from collections import Counter
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print('[WARNING] imbalanced-learn not found. pip install imbalanced-learn')

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, 'dataset')
OUTPUT_DIR = os.path.join(BASE_DIR, 'preprocessed')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_csv(p): df=pd.read_csv(p,header=None); print(f'  {os.path.basename(p)} -> {df.shape}'); return df
def report(df,n):
    print(f'\n[Quality] {n}')
    print(f'  NaN={df.isnull().sum().sum()}  Dups={df.duplicated().sum()}')
    for c,v in df.iloc[:,-1].value_counts().sort_index().items():
        print(f'  Class {int(c)}: {v} ({v/len(df)*100:.1f}%)')
def clean(df):
    n=df.isnull().sum().sum()
    if n>0: df=df.fillna(df.median(numeric_only=True))
    d=df.duplicated().sum()
    if d>0: df=df.drop_duplicates().reset_index(drop=True)
    return df
def XY(df): return df.iloc[:,:-1].values.astype('float32'), df.iloc[:,-1].values.astype('int64')
def norm(Xtr,Xte=None):
    sc=MinMaxScaler(); Xtr=sc.fit_transform(Xtr)
    return (Xtr,sc.transform(Xte),sc) if Xte is not None else (Xtr,sc)
def smote(X,y):
    if not SMOTE_AVAILABLE: return X,y
    print(f'  SMOTE before: {dict(sorted(Counter(y).items()))}')
    X,y=SMOTE(random_state=42).fit_resample(X,y)
    print(f'  SMOTE after : {dict(sorted(Counter(y).items()))}')
    return X,y
def save(name,X,y,split=''):
    tag=f'{name}_{split}' if split else name
    np.save(os.path.join(OUTPUT_DIR,f'{tag}_X.npy'),X)
    np.save(os.path.join(OUTPUT_DIR,f'{tag}_y.npy'),y)
    df=pd.DataFrame(X); df['label']=y
    df.to_csv(os.path.join(OUTPUT_DIR,f'{tag}.csv'),index=False)
    print(f'  Saved: {tag}')

print('\n=== MIT-BIH ===')
tr=clean(load_csv(os.path.join(DATA_DIR,'mitbih_train.csv')))
te=clean(load_csv(os.path.join(DATA_DIR,'mitbih_test.csv')))
report(tr,'train'); report(te,'test')
Xtr,ytr=XY(tr); Xte,yte=XY(te)
Xtr,Xte,_=norm(Xtr,Xte)
Xtr,ytr=smote(Xtr,ytr)
Xtr,ytr=shuffle(Xtr,ytr,random_state=42)
save('mitbih',Xtr,ytr,'train'); save('mitbih',Xte,yte,'test')

print('\n=== PTBDB ===')
n=load_csv(os.path.join(DATA_DIR,'ptbdb_normal.csv')); n.iloc[:,-1]=0
a=load_csv(os.path.join(DATA_DIR,'ptbdb_abnormal.csv')); a.iloc[:,-1]=1
df=clean(shuffle(pd.concat([n,a],ignore_index=True),random_state=42))
sp=int(0.8*len(df)); tr,te=df.iloc[:sp],df.iloc[sp:]
print(f'  Train:{tr.shape} Test:{te.shape}')
Xtr,ytr=XY(tr); Xte,yte=XY(te)
Xtr,Xte,_=norm(Xtr,Xte)
Xtr,ytr=smote(Xtr,ytr)
Xtr,ytr=shuffle(Xtr,ytr,random_state=42)
save('ptbdb',Xtr,ytr,'train'); save('ptbdb',Xte,yte,'test')

print('\nDone! Files in preprocessed/')
for f in sorted(os.listdir(OUTPUT_DIR)):
    sz=os.path.getsize(os.path.join(OUTPUT_DIR,f))/(1024**2)
    print(f'  {f} ({sz:.1f} MB)')