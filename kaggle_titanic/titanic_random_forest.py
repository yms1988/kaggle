import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def convert_name_to_title(df):
    ''' Convert original name to title. Here title is such as "Mr","Miss" and so on.
    '''
    df['title'] = df.Name.str.extract(' ([A-Za-z]+).', expand=False)

    df['title'] = df['title'].replace(['Lady', 'Countess','Capt', 'Col','Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['title'] = df['title'].replace('Mlle', 'Miss')
    df['title'] = df['title'].replace('Ms', 'Miss')
    df['title'] = df['title'].replace('Mme', 'Mrs')

    MAP_TITLE = {'Mr': 1, 'Miss': 2, 'Mrs': 3, 'Master': 4, 'Rare': 5}
    df['title'] = df['title'].map(MAP_TITLE)
    df['title'] = df['title'].fillna(0)

    del df['Name']


def preprocess_for_tickets(df):
    df['Ticket_Lett'] = df['Ticket'].apply(lambda x: str(x)[0])
    df['Ticket_Lett'] = df['Ticket_Lett'].apply(lambda x: str(x))
    df['Ticket_Lett'] = np.where(
        (df['Ticket_Lett']).isin(['1', '2', '3', 'S', 'P', 'C', 'A']),
        df['Ticket_Lett'],
        np.where((df['Ticket_Lett']).isin(['W', '4', '7', '6', 'L', '5', '8']),
        '0','0')
    )

    MAP_TICKET_LETTER = {'0':0,'1':1,'2':2,'3':3,'S':3,'P':0,'C':3,'A':3}
    df['Ticket_Lett']=df['Ticket_Lett'].map(MAP_TICKET_LETTER)
    df['Ticket_Len'] = df['Ticket'].apply(lambda x: len(x))

    del df['Ticket']


def preprocess_for_cabin(df):
    df['Cabin_Lett'] = df['Cabin'].apply(lambda x: str(x)[0])
    df['Cabin_Lett'] = df['Cabin_Lett'].apply(lambda x: str(x))
    df['Cabin_Lett'] = np.where(
        (df['Cabin_Lett']).isin([ 'F', 'E', 'D', 'C', 'B', 'A']),
        df['Cabin_Lett'],
        np.where((df['Cabin_Lett']).isin(['W', '4', '7', '6', 'L', '5', '8']),
        '0','0')
    )

    MAP_CABIN_LETTER = {'A':1,'B':2,'C':1,'0':0,'D':2,'E':2,'F':1}
    df['Cabin_Lett']= df['Cabin_Lett'].map(MAP_CABIN_LETTER)

    del df['Cabin']


TRAIN_DATA = './input/train.csv'
TEST_DATA = './input/test.csv'
print('\n\n\n\n\n')

# ---- Load data ----
train= pd.read_csv(TRAIN_DATA)
test= pd.read_csv(TEST_DATA)
combine = [train,test]

# ---- Preprocess ----
MAP_SEX = {'male':0, 'female':1}
MAP_EMBARKED = {'S':0, 'C':1, 'Q':2}
for dataset in combine:

    # Convert categorical numerics
    dataset['Sex'] = dataset['Sex'].map(MAP_SEX)
    dataset['Embarked'] = dataset['Embarked'].map(MAP_EMBARKED)

    # Fill luck of data
    dataset["Age"].fillna(dataset.Age.mean(), inplace=True)
    dataset["Embarked"].fillna(dataset.Embarked.mean(), inplace=True)

    # Convert non categorical features
    convert_name_to_title(dataset)
    preprocess_for_tickets(dataset)
    preprocess_for_cabin(dataset)

    # Create familysize and isAlone
    dataset["FamilySize"] = dataset["SibSp"] + dataset["Parch"] + 1
    dataset['IsAlone'] = 0
    dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1

print(test.head())

train_data = train.values
xs = train_data[:, 2:] # Pclass以降の変数
y  = train_data[:, 1]  # 正解データ

test["Age"].fillna(train.Age.mean(), inplace=True)
test["Fare"].fillna(train.Fare.mean(), inplace=True)

test_data = test.values
xs_test = test_data[:, 1:]


# --- Training ----
from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=25, max_features='auto', max_leaf_nodes=None,
            min_samples_leaf=1, min_samples_split=15,
            min_weight_fraction_leaf=0.0, n_estimators=51, n_jobs=4,
            oob_score=False, random_state=0, verbose=0, warm_start=False)

random_forest.fit(xs, y)
Y_pred = random_forest.predict(xs_test)
acc_random_forest = round(random_forest.score(xs, y) * 100, 2)
print(acc_random_forest)

submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': Y_pred
})
submission.to_csv('./output/random_forest2.csv', index=False)
