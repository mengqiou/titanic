import pandas as pd
import random

train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

def infer_deck(pclass, group_size):
    if pclass == 1:
        if group_size >= 4:
            return random.choice(['B', 'C'])  # Suites or shared cabins
        elif group_size == 3:
            return random.choice(['B', 'C', 'D'])
        elif group_size <= 2:
            return random.choice(['A', 'B', 'C', 'D'])
    elif pclass == 2:
        return random.choice(['D', 'E', 'F'])
    elif pclass == 3:
        return random.choice(['E', 'F'])
    else:
        return 'U'

##############################################################################
#####                STEP 1 fill in missing data                         #####
##############################################################################
# print(train.isnull().sum())
# print(test.isnull().sum())
train['Age'] = train.groupby(['Pclass', 'Sex'])['Age'].transform(lambda x: x.fillna(x.median()))
test['Age'] = test.groupby(['Pclass', 'Sex'])['Age'].transform(lambda x: x.fillna(x.median()))
test['Fare'] = test.groupby(['Pclass', 'Embarked'])['Fare'].transform(lambda x: x.fillna(x.median()))
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

ticket_counts = train['Ticket'].value_counts()
# Step 2: Map that count back to the original dataset
train['GroupSize'] = train['Ticket'].map(ticket_counts)

test_ticket_counts = test['Ticket'].value_counts()
test['GroupSize'] = test['Ticket'].map(test_ticket_counts)
shared_tickets = ticket_counts[ticket_counts > 1]

for df in [train, test]:
    df['Deck'] = df['Cabin'].astype(str).str[0]

# Combine both datasets if you want the full set
combined_decks = pd.concat([train['Deck'], test['Deck']], axis=0)

for df in [train, test]:
    valid_deck_rows = df[~df['Deck'].isin(['n', 'U'])]
    # apply the know deck info to the same group who share a ticket
    known_deck_map = valid_deck_rows.groupby('Ticket')['Deck'].agg(lambda x: x.mode()[0]).to_dict()
    # now for the rows where deck is unknow, we need to use:
    df['Deck'] = df.apply(
        lambda row: infer_deck(row['Pclass'], row['GroupSize']) if row['Deck'] == 'U' else row['Deck'],
        axis=1
    )

train = pd.get_dummies(train, columns=['Deck'], drop_first=True)
test = pd.get_dummies(test, columns=['Deck'], drop_first=True)

# Ensure alignment
for col in train.columns:
    if col.startswith('Deck_') and col not in test.columns:
        test[col] = 0

train.drop('Cabin', axis=1, inplace=True)
test.drop('Cabin', axis=1, inplace=True)


# women and child first policy
for df in [train, test]:
    df['IsPriority'] = 0
    df.loc[(df['Sex'] == 1) & (df['Pclass'] <= 2), 'IsPriority'] = 1  # Female in 1st or 2nd class
    df.loc[(df['Age'] < 15) & (df['Pclass'] <= 2), 'IsPriority'] = 1  # Child in 1st or 2nd class

##############################################################################
#####                STEP 3 Feature extraction                           #####
##############################################################################
features = ['Pclass', 'Sex', 'Age', 'Fare', 'GroupSize', 'IsPriority'] + \
           [col for col in train.columns if col.startswith('Title_')]
# Update features to include new Deck columns
features += [col for col in train.columns if col.startswith('Deck_')]
##############################################################################
#####                        STEP 4 Submission                           #####
##############################################################################
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.ensemble import VotingClassifier
# from sklearn.linear_model import LogisticRegression
# from xgboost import XGBClassifier


# best_rf = RandomForestClassifier(
#     n_estimators=200,  # replace with your best-found value
#     max_depth=8,       # from previous GridSearch
#     min_samples_split=4,
#     random_state=42
# )

# X = train[features]
# y = train['Survived']
# X_test = test[features]

# best_rf.fit(X, y)

# voting = VotingClassifier(estimators=[
#     ('rf', best_rf),
#     ('lr', LogisticRegression(max_iter=1000)),
#     ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
# ], voting='hard')

# voting.fit(X, y)
# predictions = voting.predict(X_test)

# submission = pd.DataFrame({
#     'PassengerId': test['PassengerId'],
#     'Survived': predictions
# })
# submission.to_csv('submission.csv', index=False)

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
submission = pd.DataFrame({
    'PassengerId': test['PassengerId'],
    'Survived': predictions
})

submission.to_csv('submission.csv', index=False)
