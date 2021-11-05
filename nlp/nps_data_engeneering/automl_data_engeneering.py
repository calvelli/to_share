#%% imports
import pandas as pd
import regex as re
from warnings import filterwarnings
filterwarnings('ignore')
import numpy as np
import spacy
from unidecode import unidecode

#python -m spacy download pt_core_news_sm

spacy.download("pt_core_news_sm")

nlp = spacy.load("pt_core_news_sm")

#%% functions
def initial_dataset_treatment(data):
    def clean_bad_data(data):
        df = data

        new_col ={
            'answer_date_bsb': 'date',
            'comment': 'review',
            'motivo': 'category',
            'submotivo': 'sub_category',
            'detractor_area': 'detractor_area'
        }
        df = df.rename(columns=new_col)

        df = df.dropna(subset=['review'])
        df = df.dropna(subset=['sub_category'])
        df = df.dropna(subset=['category'])

        df = df[['review', 'category' ,'sub_category']]

        return df

    def correct_categories(data):
        df = data

        new_arvore_sub = np.load('./data/subcategory_correction_dictionary.npy',
            allow_pickle=True).item()
        new_arvore_cat = np.load('./data/maincategory_correction_dictionary.npy',
            allow_pickle=True).item()

        df['category'] = df['sub_category'].map(new_arvore_cat)
        df['sub_category'] = df['sub_category'].map(new_arvore_sub) 


        df = df[(~df['sub_category'].isin(['-','Outros']))\
            & (df['category']!='Outros')]

        return df
       

    df = data
    df = clean_bad_data(df)
    df = correct_categories(df)

    df.drop_duplicates(subset=['review'], keep='first',inplace=True)
    df.dropna(inplace=True)

    return df


def fix_abbreviations(x):
    abbrev_dict = {
        'td':'tudo',
        'tdo':'tudo',
        'tds':'todos',
        'p':'para',
        'vc':'você',
        'vcs':'vocês',
        'obg':'obrigado',
        'tbm':'também',
        'n':'não',
        'pq': 'por que',
        'q':'que',
        'blz':'beleza',
        'tx':'taxa',
        'txs':'taxas',
        'sac':'atendimento',
        'tou':'estou',
        'to':'estou',
        'app':'aplicativo',
        'oq': 'o que',
        'hj':'hoje'
    }
    if abbrev_dict.get(x) is not None:
        return abbrev_dict.get(x)
    else:    
        return x


def remove_white_space(description):
    description = " ".join(description)
    description = " ".join(description.split())

    return description.strip().split()


def review_engeneering(data):
    def artificial_review_creation(data,variable='sub_category',count_target=200):
        def sample_review(df,count_target,categories,category,variable):

            reviews_to_create = count_target - categories[category]

            indexnames = df[df[variable]==category].index

            sample_a = np.random.choice(indexnames,reviews_to_create)
            sample_b = np.random.choice(indexnames,reviews_to_create)

            return sample_a, sample_b
        

        def create_new_review(df,sample_a,sample_b):
            lista = []
            for i1,i2 in zip(sample_a,sample_b):
                text_a = df.loc[df.index==i1,'review'].values[0]
                text_b = df.loc[df.index==i2,'review'].values[0]

                new_review = text_a+' '+text_b

                lista.append([new_review,None,cat])

            artificial_cat_data = pd.DataFrame(data=lista,columns=df.columns)

            return artificial_cat_data


        df = data
        samples = data[variable].value_counts()
        categories = samples[samples < count_target]

        if(len(categories)!=0):
            artificial_data = pd.DataFrame()

            for cat in categories.index:
                sample_a, sample_b = sample_review(df,count_target,categories,cat,variable)
                
                artificial_cat_data = create_new_review(df,sample_a,sample_b)

                artificial_data = artificial_data.append(
                    artificial_cat_data,ignore_index=True
                )
            
            df = df.append(artificial_data,ignore_index=True)

            return df             

        else:
            print('No additional data is needed!')
            return df


    def review_standardizer(text):
        review = text
        review = review.lower().replace('.',' ').replace(';',' ').replace(',',' ')

        review = [
            word.text for word in nlp(review)
            if  word.like_num is False 
            and word.is_punct is False
            and word.is_stop is False
        ]

        review = [fix_abbreviations(word) for word in review]
        review = remove_white_space(review)
        review = " ".join(review)

        review = [word.lemma_ for word in nlp(review)]
        review = " ".join(review)

        return review


    def fix_empty_categorical_feature(data):

        df = data

        sub_cats = df.loc[df['category'].isna(),'sub_category'].unique()

        for cat in sub_cats:
            category = df.loc[(df['sub_category']==cat)\
                & (~df['sub_category'].isna()),'category'].values[0]
            
            df.loc[(df['sub_category']==cat)\
                & (df['category'].isna()),'category'] = category

        return df

    
    def normalize(data,columns=['review','sub_category','category']):
        df = data
        for col in columns:
            df[col] = df[col].apply(unidecode)
        
        return df


    df = data
    df = artificial_review_creation(df,count_target=300)
    df['review'] = df['review'].apply(review_standardizer)
    
    
    #clean super short reviews
    df['len_review'] = df['review'].apply(len)
    df = df[df['len_review']>3]
    df.drop(columns='len_review',inplace=True)

    df = fix_empty_categorical_feature(df)
    df.dropna(inplace=True)

    df = normalize(df)

    return df


#%%
def main():
    df = pd.read_csv('./data/nps_new.csv')

    df = initial_dataset_treatment(df)

    df.to_csv('./data/nps_cat_fixxed_for_nlp_case.csv',index=False)

    df = review_engeneering(df)   

    df.to_csv('./data/nps_new_treated_and_lemmatized.csv',index=False)
# %%
main()
# %%
