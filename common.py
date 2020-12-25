
def explore_dataframe(df):
    print("==================")
    print("Top 5 Rows")
    print("==================")
    print(df.head(5))
    print("==================")
    print("Basics Statistics")
    print("==================")
    print(df.describe())
    print("==================")
    print("Data Types")
    print("==================")
    print(df.info())
    print("==================")
    print("Missing Values")
    print("==================")
    print(df.apply(lambda x: x.isna().sum()))
    print("==================")
    print("Histogram")
    print("==================")
    print(df.hist(figsize=(15, 12), bins=20))
    print("==================")
    print("Correlation")
    print("==================")
    print(df.corr())
    print("==================")
    print("Column Names")
    print("==================")
    print(df.columns)


def explore_none(df):
    print(df.apply(lambda x: x.isna().sum()))
