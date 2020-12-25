
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