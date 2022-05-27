import sys
from sqlalchemy import create_engine
import pandas as pd



def load_data(messages_filepath, categories_filepath):
    '''
    Reading the data and save it in as needed

    INPUT:
    path, dpath

    OUTBUT:
    dataframe
    '''
    print('starting loading\n\n\n')
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how='outer',on='id')
    print('done loading data\n\n\n')
    return df


def clean_data(df):
    '''
    cleaning the text from all other staff

    INPUT:
    dataframe

    OUTBUT:
    dataframe
    '''
    print('starting cleaning\n\n\n')
    cat = df['categories'].str.split(';', expand = True)
    row = cat.iloc[0,:]
    cat_cols = row.transform(lambda x: x[:-2]).tolist()
    cat.columns = cat_cols
    for col in cat:
        cat[col] = cat[col].transform(lambda x: x[-1:])
        cat[col] = cat[col].astype(int)
    df = df.drop('categories',axis=1)
    df = pd.concat([df, cat], axis = 1)
    df.drop('child_alone',axis=1, inplace = True)
    df = df[df['related'] != 2]
    df = df.drop_duplicates()
    print('done cleaning\n\n\n')
    return df


def save_data(df, database_filename):
    '''
    Reading the data and save it in as needed

    INPUT:
    db, database file

    OUTBUT:
    none
    '''
    print('starting saving\n\n\n')
    engine = create_engine('sqlite:///{}'.format(database_filename))
    print('creating engine\n\n\n')
    df.to_sql('df', engine, index=False, if_exists='append')
    print('complete saving\n\n\n')


    


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()