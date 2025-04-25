import pandas as pd
import random

def guess_deck_by_class(pclass):
    if pclass == 1:
        return random.choice(['A', 'B', 'C', 'D'])
    elif pclass == 2:
        return random.choice(['D', 'E', 'F'])
    elif pclass == 3:
        return random.choice(['E', 'F', 'G'])
    else:
        return 'U'


pd.set_option('display.max_rows', None)
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

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
    print(known_deck_map)

    # now for the rows where deck is unknow, we need to use:
    df['Deck'] = df.apply(
        lambda row: guess_deck_by_class(row['Pclass']) if row['Deck'] == 'U' else row['Deck'],
        axis=1
    )

# Show rows where 'Cabin' is not null
cherry = train[train['GroupSize'] >= 3]
cherry = cherry[cherry['Deck'] == 'F']
# print(cherry[['Pclass', 'Age', 'Fare', 'GroupSize', 'Cabin', 'Ticket', 'Deck']])
#....
# find out 1st class located mainly in Deck A, B, C, and some in D, E
# Deck D some 2nd class in Deck D y
# Deck E some 2nd class and 3rd class
# Deck F  -  2nd and 3rd class only

# 1st class fare : 30 + per person
# 2nd class fare:  ~12-13 dollar
# 3rd class fare: ~5-6 dollar

# if group size >= 4, only deck B and C
# if group size == 3, it could be in B, C, D, E, F
# didn't see any entry with group size >= 3 in Deck A

# Filter out unknown decks (optional)
filtered = train[train['Deck'].isin(['A', 'B', 'C', 'D', 'E', 'F'])]
deck_survival = filtered.groupby('Deck')['Survived'].mean()
# print(deck_survival)
