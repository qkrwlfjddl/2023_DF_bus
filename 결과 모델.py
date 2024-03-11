import pandas as pd

air = pd.read_csv("2022년 일별평균대기오염도_2022.csv", encoding='cp949')
weather = pd.read_csv("2022 서울시 기상데이터.csv", encoding='cp949')

print(air)
print(weather)

format = '%Y-%m-%d'
air['측정일시'] = air['측정일시'].apply(lambda x: pd.to_datetime(str(x), format= format))
weather['일시'] = weather['일시'].apply(lambda x: pd.to_datetime(str(x), format= format))

df = pd.merge(air, weather, left_on='측정일시', right_on='일시')
df.drop(['일시'],axis=1, inplace=True)
df.head()

# Calculate the median of numeric columns only
df_numeric = df.select_dtypes(include='number')
df = df.fillna(df_numeric.median())
df1 = df.isnull().sum()
print(df1)

#변수선택한 독립변수들만 추출
col = ['측정일시', '측정소명', '풍향(16방위)','이슬점온도(°C)','현지기압(hPa)','시정(10m)', '일산화탄소농도(ppm)','미세먼지농도(㎍/㎥)', '초미세먼지농도(㎍/㎥)']
df = df[col]
def change_value(x):
  if x >= 35:
    return 1
  else:
    return 0

df.iloc[:, 8] = df.iloc[:, 8].apply(change_value)
df['초미세먼지농도(㎍/㎥)'].value_counts()

#------------------------------------------------------------------

#train, test 분리
from sklearn.model_selection import train_test_split
feature = df.iloc[:,:8]
target = df['초미세먼지농도(㎍/㎥)']

X_train, X_test, y_train, y_test = train_test_split(feature, target, test_size=0.3, stratify=target, random_state=100)

#oversampling
from imblearn.over_sampling import RandomOverSampler
X_train2, y_train2 = RandomOverSampler(random_state=0).fit_resample(X_train, y_train)
X_test2, y_test2 = RandomOverSampler(random_state=0).fit_resample(X_test, y_test)

# 결정트리 모델 생성 및 훈련
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(random_state=0, max_depth=3)
tree.fit(X_train2.iloc[:,2:8], y_train2)

y_pred = tree.predict(X_test2.iloc[:,2:8])

# 모델 성능 평가
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score

auc = roc_auc_score(y_test2, y_pred)
accuracy = accuracy_score(y_test2, y_pred)
recall = recall_score(y_test2, y_pred)
precision = precision_score(y_test2, y_pred)
f1 = f1_score(y_test2, y_pred)

print("AUC:", auc)
print("Accuracy:", accuracy)
print("Recall:", recall)
print("Precision:", precision)
print("F1 스코어:", f1)

#-------------------
X_test2['pred'] = y_pred
result = X_test2['측정소명'].loc[X_test2['pred']==1].value_counts().to_frame()
print(result)
