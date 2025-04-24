import pandas as pd

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

##############################################################################
#####                STEP 1 fill in missing data                         #####
##############################################################################
print(train.isnull().sum())
print(test.isnull().sum())
train['Age'] = train.groupby(['Pclass', 'Sex'])['Age'].transform(lambda x: x.fillna(x.median()))
test['Age'] = test.groupby(['Pclass', 'Sex'])['Age'].transform(lambda x: x.fillna(x.median()))

test['Fare'] = test.groupby(['Pclass', 'Embarked'])['Fare'].transform(lambda x: x.fillna(x.median()))
# # Drop rows where 'Embarked' or 'Pclass' is missing (just in case)
# df = train.dropna(subset=['Embarked', 'Pclass'])

# # Step 1: Count total passengers per Embarked port
# total_per_port = df['Embarked'].value_counts()

# # Step 2: Count 1st-class passengers per port
# first_class_per_port = df[df['Pclass'] == 1]['Embarked'].value_counts()

# # Step 3: Compute proportions
# proportion_first_class = first_class_per_port / total_per_port

# # Step 4: Show the result, sorted by highest proportion
# proportion_first_class = proportion_first_class.sort_values(ascending=False)
# print(proportion_first_class)

train['Embarked'].fillna('C', inplace=True)

train.drop('Cabin', axis=1, inplace=True)
test.drop('Cabin', axis=1, inplace=True)


##############################################################################
#####                STEP 2 Feature engineering                         #####
##############################################################################
# Extract Title from Name
for df in [train, test]:
    df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
    df['Title'] = df['Title'].replace(['Lady', 'Countess', 'Capt', 'Col',
        'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')

# Create IsAlone feature
for df in [train, test]:
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

##############################################################################
#####                STEP 3 encode string field                          #####
##############################################################################
train = pd.get_dummies(train, columns=['Embarked'], drop_first=True)
test = pd.get_dummies(test, columns=['Embarked'], drop_first=True)

# Bug fix: ensure test set has same dummy columns as train
for col in ['Embarked_Q', 'Embarked_S']:
    if col not in test.columns:
        test[col] = 0

train['Sex'] = train['Sex'].map({'male': 0, 'female': 1})
test['Sex'] = test['Sex'].map({'male': 0, 'female': 1})

##############################################################################
#####                STEP 3 Feature extraction                           #####
##############################################################################
features = ['Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch',
            'Embarked_Q', 'Embarked_S', 'IsAlone'] + \
           [col for col in train.columns if col.startswith('Title_')]

from sklearn.ensemble import RandomForestClassifier

X = train[features]
y = train['Survived']

X_test = test[features]

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    min_samples_split=4,
    random_state=42
)

model.fit(X, y)

predictions = model.predict(X_test)
##############################################################################
#####                        STEP 4 Submission                           #####
##############################################################################
submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': predictions
})

submission.to_csv('submission.csv', index=False)



