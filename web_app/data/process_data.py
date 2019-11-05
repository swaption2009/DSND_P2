import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath, index_col='id')
    categories = pd.read_csv(categories_filepath, index_col='id')
    return messages.join(categories, how='left')


def clean_data(df):
    x = df['categories'].str.split(";", n=len(df['categories'].str.split(";").iloc[0]))
    tmp_x = pd.DataFrame(x.values.tolist(), index= x.index)
    
    row = tmp_x.iloc[0]
    tmp_x_colnames = [x[:-2] for x in row]
    tmp_x.columns = tmp_x_colnames

    for column in tmp_x:
      tmp_x[column] = tmp_x[column].apply(lambda x: x[-1])
      tmp_x[column] = tmp_x[column].astype(int)

    df = df.drop(['categories'], axis=1)
    df = df.join(tmp_x)
    df = df.drop_duplicates(subset='message', keep='first')

    return df


def save_data(df, database_filename):
  database_engine = 'sqlite:///' + database_filename
  engine = create_engine(database_engine)
  df.to_sql('InsertTableName', engine, index=False)


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