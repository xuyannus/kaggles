
def age_encoding(x):
    if x <= 15:
        return 1
    elif x <= 60:
        return 2
    else:
        return 3


def sex_encoding(x):
    if x == 'female':
        return 1
    elif x == 'male':
        return 2


# https://www.kaggle.com/gunesevitan/titanic-advanced-feature-engineering-tutorial
cabin_map = {
    'A': 'ABCT',
    'B': 'ABCT',
    'C': 'ABCT',
    'T': 'ABCT',
    'D': 'DE',
    'E': 'DE',
    'F': 'FG',
    'F': 'FG',
    'M': 'M',
}


def cabin_encoding(x):
    return cabin_map[x] if x in cabin_map else 'M'


def family_size_encoding(x):
    if x <= 1:
        return 1
    elif x <= 4:
        return 2
    elif x <= 6:
        return 3
    else:
        return 4


def fare_encoding(x):
    if x <= 5:
        return 1
    elif x <= 10:
        return 2
    elif x <= 30:
        return 3
    else:
        return 4
