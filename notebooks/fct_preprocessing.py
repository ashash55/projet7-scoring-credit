# Fonctions de preprocessing des datasets

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import LabelEncoder
import gc
from contextlib import contextmanager
import time


# --------------------------------------------------------------------------------
# ------------------ FONCTIONS ANALYSE EXPLORATOIRE ------------------------------
# --------------------------------------------------------------------------------

def tx_rempl_min(df, tx_remplissage_min=70):
    """Fonction qui renvoie le dataframe supprimé des variables
    qui sont sous le seuil de remplissage indiqué en entrée et affiche
    le taux de remplissage des variables restantes qui contiennent des valeurs
    manquantes, le nombre de NaN ainsi que leur dtype

    Arguments:
    --------------------------------
    df: dataframe: tableau en entrée, obligatoire
    tx_remplissage_min : int : seuil de remplissage minimum des variables,
    70% par défaut soit 30% de données manquantes


    return:
    --------------------------------
    dataframe dont chaque variable est supérieure au seuil
    de remplissage indiqué et le type de la variable"""

    df_clean = df.loc[:, df.isna().mean() <= ((100 - tx_remplissage_min) / 100)]

    return df_clean


def iterative_imputer_function(df_var_corr):
    """Fonction qui impute les valeurs manquantes en fonction
    des données des variables corrélées entre-elles

    Arguments:
    --------------------------------
    df_var_corr: dataframe: tableau en entrée ne comportant que les variables
    corrélées entre-elles, obligatoire

    return:
    --------------------------------
    array avec les données manquantes estimées par iterative
    imputer"""

    imp_mean = IterativeImputer(random_state=42)
    x = np.array(df_var_corr)
    imp_mean.fit(x)
    result = imp_mean.transform(x)

    return result


def nan_a_retraiter(df):
    """Fonction qui affiche le taux de remplissage des variables restantes
    qui contiennent des valeurs manquantes, le nombre de NaN ainsi que leur dtype

    Arguments:
    --------------------------------
    df: dataframe: tableau en entrée, obligatoire

    return:
    --------------------------------
    Dataframe des variables restant à retraiter"""

    # print('Taux de remplissage des variables restantes à retraiter :')
    not_nan_percent = round(df.notna().mean().sort_values(ascending=False), 4) * 100
    nan_nb = round(df.isna().sum().sort_values(ascending=False), 4)
    dtype = df.dtypes
    df_concat = pd.concat([not_nan_percent, nan_nb, dtype], keys=['Tx_rempl', 'nb_NaN', 'dtypes'], axis=1)
    df_concat = df_concat[df_concat['nb_NaN'] > 0]

    return df_concat


def categories_encoder(df, nan_as_category=True):
    """Fonction de preprocessing des variables catégorielles. Applique un
    One Hot Encoder sur les variables catégorielles non binaires et un Label
    Encoder pour les variables catégorielles binaires.

    Arguments:
    --------------------------------
    df: dataframe: tableau en entrée, obligatoire
    nan_as_category : bool, considère les valeurs manquantes comme une catégorie
    à part entière. Vrai par défaut.

    return:
    --------------------------------
    None
    """

    df_columns = list(df.columns)

    # Colonnes pour OHE (modalités > 2)
    categ_columns_ohe = [col for col in df.columns if df[col].dtype == 'object']
    df_ohe = df[categ_columns_ohe]
    categ_columns_ohe = [col for col in df_ohe.columns if len(list(df_ohe[col].unique())) > 2]

    # Colonnes pour Label Encoder (modalités <= 2)
    categ_columns_le = [col for col in df.columns if df[col].dtype == 'object']
    df_le = df[categ_columns_le]
    categ_columns_le = [col for col in df_le.columns if len(list(df_ohe[col].unique())) <= 2]

    # Label encoder quand modalités <= 2
    le = LabelEncoder()
    for col in df[categ_columns_le]:
        le.fit(df[col])
        df[col] = le.transform(df[col])

    # One Hot Encoder quand modalités > 2
    df = pd.get_dummies(df, columns=categ_columns_ohe, dummy_na=nan_as_category)
    new_columns = [c for c in df.columns if c not in df_columns] + categ_columns_le
    return df, new_columns


# --------------------------------------------------------------------------------
# ---------------------- PREPROCESSING DATASETS ----------------------------------
# --------------------------------------------------------------------------------


def preprocess_app_train():
    """Fonction de preprocessing du dataset application_train :
        traitement des données manquantes, traitement des outliers et des modalités,
        feature engineering

        Arguments:
        --------------------------------


        return:
        --------------------------------
        Dataframe preprocessé"""

    # Chargement du dataset

    # Get the absolute path to the data directory
    script_path = Path(__file__).resolve()
    data_path = script_path.parent / 'data' / 'application_train.csv'
    app_train = pd.read_csv(data_path, sep=',', encoding='utf-8')

    # Traitement des données manquantes

    # Suppression des variables avec 30% ou plus de NaN
    app_train = tx_rempl_min(app_train, 70)

    # iterative imputer pour les variables corrélées entre-elles
    app_train['OBS_30_CNT_SOCIAL_CIRCLE'] = iterative_imputer_function(
        app_train[['OBS_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE']])[:, 0]
    app_train['OBS_60_CNT_SOCIAL_CIRCLE'] = iterative_imputer_function(
        app_train[['OBS_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE']])[:, 1]
    app_train['AMT_GOODS_PRICE'] = iterative_imputer_function(
        app_train[['AMT_GOODS_PRICE', 'AMT_CREDIT']])[:, 0]
    app_train['CNT_FAM_MEMBERS'] = iterative_imputer_function(
        app_train[['CNT_FAM_MEMBERS', 'CNT_CHILDREN']])[:, 0]
    app_train['DEF_60_CNT_SOCIAL_CIRCLE'] = iterative_imputer_function(
        app_train[['DEF_60_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE']])[:, 0]
    app_train['DEF_30_CNT_SOCIAL_CIRCLE'] = iterative_imputer_function(
        app_train[['DEF_60_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE']])[:, 1]
    app_train['AMT_GOODS_PRICE'] = iterative_imputer_function(
        app_train[['AMT_GOODS_PRICE', 'AMT_ANNUITY']])[:, 0]
    app_train['AMT_ANNUITY'] = iterative_imputer_function(
        app_train[['AMT_GOODS_PRICE', 'AMT_ANNUITY']])[:, 1]
    app_train['AMT_ANNUITY'] = iterative_imputer_function(
        app_train[['AMT_CREDIT', 'AMT_ANNUITY']])[:, 1]

    # Médiane et mode pour variables avec taux de NaN > 80% (hors variable TARGET)
    df_nan = nan_a_retraiter(app_train)
    col_med = df_nan[(df_nan['Tx_rempl'] > 80) &
                     (df_nan['dtypes'] == 'float64') &
                     (~ df_nan.index.str.contains('TARGET'))].index.tolist()
    for col in col_med:
        app_train[col] = app_train[col].fillna(app_train[col].median())
    del app_train['AMT_REQ_CREDIT_BUREAU_YEAR']
    del app_train['EXT_SOURCE_3']
    app_train['NAME_TYPE_SUITE'] = app_train['NAME_TYPE_SUITE'].fillna(app_train['NAME_TYPE_SUITE'].mode()[0])

    # Traitements des variables quantitatives

    # Nombre d'enfants
    app_train.loc[app_train['CNT_CHILDREN'] == 0, 'CNT_CHILDREN'] = 'No child'
    app_train.loc[app_train['CNT_CHILDREN'] == 1, 'CNT_CHILDREN'] = 'One child'
    app_train.loc[app_train['CNT_CHILDREN'] == 2, 'CNT_CHILDREN'] = 'Two children'
    app_train['CNT_CHILDREN'] = np.where(~app_train['CNT_CHILDREN'].isin(['No child', 'One child', 'Two children']),
                                         '> Three children', app_train['CNT_CHILDREN'])

    # Nombre de membres dans la famille
    app_train = app_train[~((app_train['CNT_FAM_MEMBERS'] < 2) & (app_train['CNT_FAM_MEMBERS'] > 1))]
    app_train.loc[app_train['CNT_FAM_MEMBERS'] == 1, 'CNT_FAM_MEMBERS_RANGE'] = '1 pers'
    app_train.loc[app_train['CNT_FAM_MEMBERS'] == 2, 'CNT_FAM_MEMBERS_RANGE'] = '2 pers'
    app_train.loc[app_train['CNT_FAM_MEMBERS'] == 3, 'CNT_FAM_MEMBERS_RANGE'] = '3 pers'
    app_train['CNT_FAM_MEMBERS_RANGE'] = np.where(~app_train['CNT_FAM_MEMBERS_RANGE'].isin(['1 pers', '2 pers',
                                                                                            '3 pers']), '> 4 pers',
                                                  app_train['CNT_FAM_MEMBERS_RANGE'])

    # Age du client
    app_train['AGE'] = round(app_train['DAYS_BIRTH'] / 365 * -1, 0)
    del app_train['DAYS_BIRTH']
    app_train.loc[app_train['AGE'] <= 30, 'AGE_RANGE'] = '< 30 YEARS OLD'
    app_train.loc[(app_train['AGE'] > 30) & (app_train['AGE'] <= 50), 'AGE_RANGE'] = 'BETWEEN 31 AND 50 YEARS OLD'
    app_train.loc[app_train['AGE'] > 50, 'AGE_RANGE'] = '> 50 YEARS OLD'

    # Ancienneté du client à son poste actuel
    app_train['YEARS_EMPLOYED'] = round(app_train['DAYS_EMPLOYED'] / 365 * -1, 2)
    del app_train['DAYS_EMPLOYED']
    # On affecte la valeur -1 aux retraités et sans emploi
    app_train.loc[app_train['YEARS_EMPLOYED'] == -1000.67, 'YEARS_EMPLOYED'] = -1
    app_train.loc[app_train['YEARS_EMPLOYED'] <= 0, 'YEARS_EMPLOYED_RANGE'] = 'Pensioner/Unemployed'
    app_train.loc[(app_train['YEARS_EMPLOYED'] > 0) &
                  (app_train['YEARS_EMPLOYED'] <= 3), 'YEARS_EMPLOYED_RANGE'] = '3 years or less'
    app_train.loc[(app_train['YEARS_EMPLOYED'] > 3) &
                  (app_train['YEARS_EMPLOYED'] <= 8), 'YEARS_EMPLOYED_RANGE'] = 'Between 3 and 8 years'
    app_train.loc[app_train['YEARS_EMPLOYED'] > 8, 'YEARS_EMPLOYED_RANGE'] = 'More than 8 years'

    # Nb années avant la demande où le client a modifié son enregistrement
    app_train['YEARS_REGISTRATION'] = round(app_train['DAYS_REGISTRATION'] / 365 * -1, 0)
    del app_train['DAYS_REGISTRATION']
    app_train.loc[app_train['YEARS_REGISTRATION'] <= 6, 'YEARS_REGISTRATION_RANGE'] = '< 6 YEARS'
    app_train.loc[(app_train['YEARS_REGISTRATION'] > 6) &
                  (app_train['YEARS_REGISTRATION'] <= 12), 'YEARS_REGISTRATION_RANGE'] = 'BETWEEN 6 AND 12 YEARS'
    app_train.loc[(app_train['YEARS_REGISTRATION'] > 12) &
                  (app_train['YEARS_REGISTRATION'] <= 20), 'YEARS_REGISTRATION_RANGE'] = 'BETWEEN 13 AND 20 YEARS'
    app_train.loc[app_train['YEARS_REGISTRATION'] > 20, 'YEARS_REGISTRATION_RANGE'] = '> 20 YEARS'

    # Heure où le client a demandé son prêt
    app_train.loc[app_train['HOUR_APPR_PROCESS_START'] < 8, 'HOUR_APPR_PROCESS_START_RANGE'] = 'Evening/Night'
    app_train.loc[(app_train['HOUR_APPR_PROCESS_START'] >= 8) &
                  (app_train['HOUR_APPR_PROCESS_START'] <= 12), 'HOUR_APPR_PROCESS_START_RANGE'] = 'Morning'
    app_train.loc[(app_train['HOUR_APPR_PROCESS_START'] > 12) &
                  (app_train['HOUR_APPR_PROCESS_START'] <= 14), 'HOUR_APPR_PROCESS_START_RANGE'] = 'Lunch Break'
    app_train.loc[(app_train['HOUR_APPR_PROCESS_START'] > 14) &
                  (app_train['HOUR_APPR_PROCESS_START'] <= 19), 'HOUR_APPR_PROCESS_START_RANGE'] = 'Afternoon'
    app_train.loc[app_train['HOUR_APPR_PROCESS_START'] > 19, 'HOUR_APPR_PROCESS_START_RANGE'] = 'Evening/Night'

    # Nb années avant la demande où le client a changé son document d'identité
    app_train['YEARS_ID_PUBLISH'] = round(app_train['DAYS_ID_PUBLISH'] / 365 * -1, 2)
    del app_train['DAYS_ID_PUBLISH']
    app_train.loc[app_train['YEARS_ID_PUBLISH'] <= 5, 'YEARS_ID_PUBLISH_RANGE'] = '< 5 YEARS'
    app_train.loc[(app_train['YEARS_ID_PUBLISH'] > 5) &
                  (app_train['YEARS_ID_PUBLISH'] <= 9), 'YEARS_ID_PUBLISH_RANGE'] = 'BETWEEN 5 AND 9 YEARS'
    app_train.loc[(app_train['YEARS_ID_PUBLISH'] > 9) &
                  (app_train['YEARS_ID_PUBLISH'] <= 12), 'YEARS_ID_PUBLISH_RANGE'] = 'BETWEEN 10 AND 12 YEARS'
    app_train.loc[app_train['YEARS_ID_PUBLISH'] > 12, 'YEARS_ID_PUBLISH_RANGE'] = '> 12 YEARS'

    # Nb années avant la demande où le client a changé son numéro de téléphone
    app_train['YEARS_LAST_PHONE_CHANGE'] = round(app_train['DAYS_LAST_PHONE_CHANGE'] / 365 * -1, 2)
    del app_train['DAYS_LAST_PHONE_CHANGE']
    app_train.loc[app_train['YEARS_LAST_PHONE_CHANGE'] <= 1, 'YEARS_LAST_PHONE_CHANGE_RANGE'] = '< 1 YEAR'
    app_train.loc[(app_train['YEARS_LAST_PHONE_CHANGE'] > 1) &
                  (app_train['YEARS_LAST_PHONE_CHANGE'] <= 2), 'YEARS_LAST_PHONE_CHANGE_RANGE'] = 'BETWEEN 1 & 2 YEARS'
    app_train.loc[(app_train['YEARS_LAST_PHONE_CHANGE'] > 2) &
                  (app_train['YEARS_LAST_PHONE_CHANGE'] <= 4), 'YEARS_LAST_PHONE_CHANGE_RANGE'] = 'BETWEEN 2 & 4 YEARS'
    app_train.loc[app_train['YEARS_LAST_PHONE_CHANGE'] > 4, 'YEARS_LAST_PHONE_CHANGE_RANGE'] = '> 4 YEARS'

    # Traitements des variables qualitatives

    # Genre
    app_train = app_train[app_train['CODE_GENDER'].isin(["F", "M"])]

    # Accompagnateur
    dico_name_type_suite = {'Unaccompanied': 'Unaccompanied',
                            'Family': 'Family',
                            'Spouse, partner': 'Family',
                            'Children': 'Family',
                            'Other_B': 'Other',
                            'Other_A': 'Other',
                            'Group of people': 'Other'}
    app_train['NAME_TYPE_SUITE'] = app_train['NAME_TYPE_SUITE'].map(dico_name_type_suite)

    # Situation professionnelle
    dico_name_income_type = {'Working': 'Private worker',
                             'Commercial associate': 'Commercial associate',
                             'Pensioner': 'Pensioner/Student/Unemployed',
                             'State servant': 'Public worker',
                             'Unemployed': 'Pensioner/Student/Unemployed',
                             'Student': 'Pensioner/Student/Unemployed',
                             'Businessman': 'Other',
                             'Maternity leave': 'Pensioner/Student/Unemployed'}
    app_train['NAME_INCOME_TYPE'] = app_train['NAME_INCOME_TYPE'].map(dico_name_income_type)

    # Education
    dico_name_education_type = {'Secondary / secondary special': 'Lower Secondary & Secondary',
                                'Higher education': 'Higher education',
                                'Incomplete higher': 'Incomplete higher',
                                'Lower secondary': 'Lower Secondary & Secondary',
                                'Academic degree': 'Higher education', }
    app_train['NAME_EDUCATION_TYPE'] = app_train['NAME_EDUCATION_TYPE'].map(dico_name_education_type)
    app_train = app_train[~app_train['NAME_FAMILY_STATUS'].isin(["Unknown"])]

    # Situation familiale
    dico_name_family_status = {'Married': 'Married',
                               'Single / not married': 'Not Married',
                               'Civil marriage': 'Married',
                               'Separated': 'Separated',
                               'Widow': 'Widow', }
    app_train['NAME_FAMILY_STATUS'] = app_train['NAME_FAMILY_STATUS'].map(dico_name_family_status)

    # Logement
    dico_name_housing_type = {'House / apartment': 'Owner',
                              'With parents': 'No or low rent',
                              'Municipal apartment': 'No or low rent',
                              'Rented apartment': 'Tenant',
                              'Office apartment': 'No or low rent',
                              'Co-op apartment': 'Tenant'}
    app_train['NAME_HOUSING_TYPE'] = app_train['NAME_HOUSING_TYPE'].map(dico_name_housing_type)

    # Moment où le client a fait la demande
    dico_weekday_process = {'TUESDAY': 'working days',
                            'WEDNESDAY': 'working days',
                            'MONDAY': 'working days',
                            'THURSDAY': 'working days',
                            'FRIDAY': 'working days',
                            'SATURDAY': 'week end',
                            'SUNDAY': 'week end'}
    app_train['WEEKDAY_APPR_PROCESS_START'] = app_train['WEEKDAY_APPR_PROCESS_START'].map(dico_weekday_process)

    # Secteur d'activité
    dico_orga_type = {'Industry: type 6': 'Industry',
                      'Industry: type 13': 'Industry',
                      'Industry: type 8': 'Industry',
                      'Industry: type 5': 'Industry',
                      'Industry: type 10': 'Industry',
                      'Industry: type 12': 'Industry',
                      'Industry: type 2': 'Industry',
                      'Industry: type 9': 'Industry',
                      'Industry: type 3': 'Industry',
                      'Industry: type 7': 'Industry',
                      'Industry: type 1': 'Industry',
                      'Industry: type 4': 'Industry',
                      'Industry: type 11': 'Industry',
                      'Business Entity Type 3': 'Business Entity',
                      'Business Entity Type 2': 'Business Entity',
                      'Business Entity Type 1': 'Business Entity',
                      'Transport: type 1': 'Transport',
                      'Transport: type 2': 'Transport',
                      'Transport: type 3': 'Transport',
                      'Transport: type 4': 'Transport',
                      'Trade: type 7': 'Trade',
                      'Trade: type 4': 'Trade',
                      'Trade: type 5': 'Trade',
                      'Trade: type 1': 'Trade',
                      'Trade: type 6': 'Trade',
                      'Trade: type 3': 'Trade',
                      'Trade: type 2': 'Trade',
                      'Bank': 'Bank/Insurance',
                      'Insurance': 'Bank/Insurance',
                      'School': 'School/University/Kindergarten',
                      'University': 'School/University/Kindergarten',
                      'Kindergarten': 'School/University/Kindergarten',
                      'Government': 'Government/Military/Security Ministries',
                      'Military': 'Government/Military/Security Ministries',
                      'Security Ministries': 'Government/Military/Security Ministries',
                      'Telecom': 'Telecom',
                      'Mobile': 'Telecom',
                      'Electricity': 'Electricity/Construction',
                      'Construction': 'Electricity/Construction',
                      'Services': 'Services',
                      'Advertising': 'Services',
                      'Legal Services': 'Services',
                      'Hotel': 'Hotel/Restaurant',
                      'Restaurant': 'Hotel/Restaurant',
                      'Cleaning': 'Cleaning/Housing',
                      'Housing': 'Cleaning/Housing',
                      'Medicine': 'Medical',
                      'Emergency': 'Medical',
                      'Security': 'Police/Security',
                      'Police': 'Police/Security',
                      'Religion': 'Other',
                      'Other': 'Other',
                      'XNA': 'Other',
                      'Self-employed': 'Self-employed',
                      'Postal': 'Postal',
                      'Agriculture': 'Agriculture',
                      'Culture': 'Culture',
                      'Realtor': 'Realtor', }
    app_train['ORGANIZATION_TYPE'] = app_train['ORGANIZATION_TYPE'].map(dico_orga_type)

    # Variables à supprimer
    var_quanti_a_suppr = ['AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_DAY',
                          'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_QRT',
                          'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE']
    var_quali_a_suppr = ['FLAG_MOBIL', 'FLAG_CONT_MOBILE', 'FLAG_EMAIL', 'WEEKDAY_APPR_PROCESS_START',
                         'LIVE_REGION_NOT_WORK_REGION', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_7',
                         'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_19',
                         'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21']
    var_suppr = var_quanti_a_suppr + var_quali_a_suppr
    app_train = app_train.drop(var_suppr, axis=1)

    # Feature engineering

    app_train['CREDIT_DURATION'] = app_train['AMT_CREDIT'] / app_train['AMT_ANNUITY']
    app_train['DEBT_RATIO'] = app_train['AMT_ANNUITY'] / app_train['AMT_INCOME_TOTAL']
    app_train['PAYMENT_RATE'] = app_train['AMT_ANNUITY'] / app_train['AMT_CREDIT']
    app_train['INCOME_PER_PERSON'] = app_train['AMT_INCOME_TOTAL'] / app_train['CNT_FAM_MEMBERS']
    app_train['INCOME_CREDIT_PERC'] = app_train['AMT_INCOME_TOTAL'] / app_train['AMT_CREDIT']
    app_train['CREDIT_GOODS_PERC'] = app_train['AMT_CREDIT'] / app_train['AMT_GOODS_PRICE']

    # Durée du crédit
    app_train.loc[app_train['CREDIT_DURATION'] < 15, 'CREDIT_DURATION_RANGE'] = 'SHORT < 15 Years'
    app_train.loc[(app_train['CREDIT_DURATION'] >= 15) &
                  (app_train['CREDIT_DURATION'] <= 20), 'CREDIT_DURATION_RANGE'] = 'MEDIUM BETWEEN 15 & 20 Years'
    app_train.loc[app_train['CREDIT_DURATION'] > 20, 'CREDIT_DURATION_RANGE'] = 'LONG > 20 Years'

    # Taux d'endettement
    app_train.loc[app_train['DEBT_RATIO'] < 0.15, 'DEBT_RATIO_RANGE'] = 'VERY LOW < 15'
    app_train.loc[(app_train['DEBT_RATIO'] >= 0.15) &
                  (app_train['DEBT_RATIO'] <= 0.20), 'DEBT_RATIO_RANGE'] = 'LOW BETWEEN 15 & 20 Years'
    app_train.loc[(app_train['DEBT_RATIO'] > 0.20) &
                  (app_train['DEBT_RATIO'] <= 0.35), 'DEBT_RATIO_RANGE'] = 'MEDIUM BETWEEN 21 & 35'
    app_train.loc[app_train['DEBT_RATIO'] > 0.35, 'DEBT_RATIO_RANGE'] = 'VERY HIGH > 35%'

    # Revenus par personne
    app_train.loc[app_train['INCOME_PER_PERSON'] < 45000, 'INCOME_PER_PERSON_RANGE'] = 'LOW < 45K'
    app_train.loc[(app_train['INCOME_PER_PERSON'] >= 45000) &
                  (app_train['INCOME_PER_PERSON'] <= 75000), 'INCOME_PER_PERSON_RANGE'] = 'MEDIUM BETWEEN 45K & 75K'
    app_train.loc[app_train['INCOME_PER_PERSON'] > 75000, 'INCOME_PER_PERSON_RANGE'] = 'HIGH > 75K'
    app_train = app_train.set_index('SK_ID_CURR')

    del var_quanti_a_suppr, var_quali_a_suppr, var_suppr
    gc.collect()
    return app_train


def preprocess_app_test():
    """Fonction de preprocessing du dataset application_test (identique à application_train) :
        traitement des données manquantes, traitement des outliers et des modalités,
        feature engineering. Ce dataset sera utilisé lors de l'analyse du data drift.

        Arguments:
        --------------------------------


        return:
        --------------------------------
        Dataframe preprocessé"""

    # Chargement du dataset

    # Get the absolute path to the data directory
    script_path = Path(__file__).resolve()
    data_path = script_path.parent / 'data' / 'application_test.csv'
    app_test = pd.read_csv(data_path, sep=',', encoding='utf-8')

    # Traitement des données manquantes

    # Suppression des variables avec 30% ou plus de NaN
    app_test = tx_rempl_min(app_test, 70)

    # iterative imputer pour les variables corrélées entre-elles
    app_test['OBS_30_CNT_SOCIAL_CIRCLE'] = iterative_imputer_function(
        app_test[['OBS_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE']])[:, 0]
    app_test['OBS_60_CNT_SOCIAL_CIRCLE'] = iterative_imputer_function(
        app_test[['OBS_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE']])[:, 1]
    app_test['AMT_GOODS_PRICE'] = iterative_imputer_function(
        app_test[['AMT_GOODS_PRICE', 'AMT_CREDIT']])[:, 0]
    app_test['CNT_FAM_MEMBERS'] = iterative_imputer_function(
        app_test[['CNT_FAM_MEMBERS', 'CNT_CHILDREN']])[:, 0]
    app_test['DEF_60_CNT_SOCIAL_CIRCLE'] = iterative_imputer_function(
        app_test[['DEF_60_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE']])[:, 0]
    app_test['DEF_30_CNT_SOCIAL_CIRCLE'] = iterative_imputer_function(
        app_test[['DEF_60_CNT_SOCIAL_CIRCLE', 'DEF_30_CNT_SOCIAL_CIRCLE']])[:, 1]
    app_test['AMT_GOODS_PRICE'] = iterative_imputer_function(
        app_test[['AMT_GOODS_PRICE', 'AMT_ANNUITY']])[:, 0]
    app_test['AMT_ANNUITY'] = iterative_imputer_function(
        app_test[['AMT_GOODS_PRICE', 'AMT_ANNUITY']])[:, 1]
    app_test['AMT_ANNUITY'] = iterative_imputer_function(
        app_test[['AMT_CREDIT', 'AMT_ANNUITY']])[:, 1]

    # Médiane et mode pour variables avec taux de NaN > 80% (hors variable TARGET)
    df_nan = nan_a_retraiter(app_test)
    col_med = df_nan[(df_nan['Tx_rempl'] > 80) &
                     (df_nan['dtypes'] == 'float64') &
                     (~ df_nan.index.str.contains('TARGET'))].index.tolist()
    for col in col_med:
        app_test[col] = app_test[col].fillna(app_test[col].median())
    del app_test['AMT_REQ_CREDIT_BUREAU_YEAR']
    del app_test['EXT_SOURCE_3']
    app_test['NAME_TYPE_SUITE'] = app_test['NAME_TYPE_SUITE'].fillna(app_test['NAME_TYPE_SUITE'].mode()[0])

    # Traitements des variables quantitatives

    # Nombre d'enfants
    app_test.loc[app_test['CNT_CHILDREN'] == 0, 'CNT_CHILDREN'] = 'No child'
    app_test.loc[app_test['CNT_CHILDREN'] == 1, 'CNT_CHILDREN'] = 'One child'
    app_test.loc[app_test['CNT_CHILDREN'] == 2, 'CNT_CHILDREN'] = 'Two children'
    app_test['CNT_CHILDREN'] = np.where(~app_test['CNT_CHILDREN'].isin(['No child', 'One child', 'Two children']),
                                        '> Three children', app_test['CNT_CHILDREN'])

    # Nombre de membres dans la famille
    app_test = app_test[~((app_test['CNT_FAM_MEMBERS'] < 2) & (app_test['CNT_FAM_MEMBERS'] > 1))]
    app_test.loc[app_test['CNT_FAM_MEMBERS'] == 1, 'CNT_FAM_MEMBERS_RANGE'] = '1 pers'
    app_test.loc[app_test['CNT_FAM_MEMBERS'] == 2, 'CNT_FAM_MEMBERS_RANGE'] = '2 pers'
    app_test.loc[app_test['CNT_FAM_MEMBERS'] == 3, 'CNT_FAM_MEMBERS_RANGE'] = '3 pers'
    app_test['CNT_FAM_MEMBERS_RANGE'] = np.where(~app_test['CNT_FAM_MEMBERS_RANGE'].isin(['1 pers', '2 pers',
                                                                                          '3 pers']), '> 4 pers',
                                                 app_test['CNT_FAM_MEMBERS_RANGE'])

    # Age du client
    app_test['AGE'] = round(app_test['DAYS_BIRTH'] / 365 * -1, 0)
    del app_test['DAYS_BIRTH']
    app_test.loc[app_test['AGE'] <= 30, 'AGE_RANGE'] = '< 30 YEARS OLD'
    app_test.loc[(app_test['AGE'] > 30) & (app_test['AGE'] <= 50), 'AGE_RANGE'] = 'BETWEEN 31 AND 50 YEARS OLD'
    app_test.loc[app_test['AGE'] > 50, 'AGE_RANGE'] = '> 50 YEARS OLD'

    # Ancienneté du client à son poste actuel
    app_test['YEARS_EMPLOYED'] = round(app_test['DAYS_EMPLOYED'] / 365 * -1, 2)
    del app_test['DAYS_EMPLOYED']
    # On affecte la valeur -1 aux retraités et sans emploi
    app_test.loc[app_test['YEARS_EMPLOYED'] == -1000.67, 'YEARS_EMPLOYED'] = -1
    app_test.loc[app_test['YEARS_EMPLOYED'] <= 0, 'YEARS_EMPLOYED_RANGE'] = 'Pensioner/Unemployed'
    app_test.loc[(app_test['YEARS_EMPLOYED'] > 0) &
                 (app_test['YEARS_EMPLOYED'] <= 3), 'YEARS_EMPLOYED_RANGE'] = '3 years or less'
    app_test.loc[(app_test['YEARS_EMPLOYED'] > 3) &
                 (app_test['YEARS_EMPLOYED'] <= 8), 'YEARS_EMPLOYED_RANGE'] = 'Between 3 and 8 years'
    app_test.loc[app_test['YEARS_EMPLOYED'] > 8, 'YEARS_EMPLOYED_RANGE'] = 'More than 8 years'

    # Nb années avant la demande où le client a modifié son enregistrement
    app_test['YEARS_REGISTRATION'] = round(app_test['DAYS_REGISTRATION'] / 365 * -1, 0)
    del app_test['DAYS_REGISTRATION']
    app_test.loc[app_test['YEARS_REGISTRATION'] <= 6, 'YEARS_REGISTRATION_RANGE'] = '< 6 YEARS'
    app_test.loc[(app_test['YEARS_REGISTRATION'] > 6) &
                 (app_test['YEARS_REGISTRATION'] <= 12), 'YEARS_REGISTRATION_RANGE'] = 'BETWEEN 6 AND 12 YEARS'
    app_test.loc[(app_test['YEARS_REGISTRATION'] > 12) &
                 (app_test['YEARS_REGISTRATION'] <= 20), 'YEARS_REGISTRATION_RANGE'] = 'BETWEEN 13 AND 20 YEARS'
    app_test.loc[app_test['YEARS_REGISTRATION'] > 20, 'YEARS_REGISTRATION_RANGE'] = '> 20 YEARS'

    # Heure où le client a demandé son prêt
    app_test.loc[app_test['HOUR_APPR_PROCESS_START'] < 8, 'HOUR_APPR_PROCESS_START_RANGE'] = 'Evening/Night'
    app_test.loc[(app_test['HOUR_APPR_PROCESS_START'] >= 8) &
                 (app_test['HOUR_APPR_PROCESS_START'] <= 12), 'HOUR_APPR_PROCESS_START_RANGE'] = 'Morning'
    app_test.loc[(app_test['HOUR_APPR_PROCESS_START'] > 12) &
                 (app_test['HOUR_APPR_PROCESS_START'] <= 14), 'HOUR_APPR_PROCESS_START_RANGE'] = 'Lunch Break'
    app_test.loc[(app_test['HOUR_APPR_PROCESS_START'] > 14) &
                 (app_test['HOUR_APPR_PROCESS_START'] <= 19), 'HOUR_APPR_PROCESS_START_RANGE'] = 'Afternoon'
    app_test.loc[app_test['HOUR_APPR_PROCESS_START'] > 19, 'HOUR_APPR_PROCESS_START_RANGE'] = 'Evening/Night'

    # Nb années avant la demande où le client a changé son document d'identité
    app_test['YEARS_ID_PUBLISH'] = round(app_test['DAYS_ID_PUBLISH'] / 365 * -1, 2)
    del app_test['DAYS_ID_PUBLISH']
    app_test.loc[app_test['YEARS_ID_PUBLISH'] <= 5, 'YEARS_ID_PUBLISH_RANGE'] = '< 5 YEARS'
    app_test.loc[(app_test['YEARS_ID_PUBLISH'] > 5) &
                 (app_test['YEARS_ID_PUBLISH'] <= 9), 'YEARS_ID_PUBLISH_RANGE'] = 'BETWEEN 5 AND 9 YEARS'
    app_test.loc[(app_test['YEARS_ID_PUBLISH'] > 9) &
                 (app_test['YEARS_ID_PUBLISH'] <= 12), 'YEARS_ID_PUBLISH_RANGE'] = 'BETWEEN 10 AND 12 YEARS'
    app_test.loc[app_test['YEARS_ID_PUBLISH'] > 12, 'YEARS_ID_PUBLISH_RANGE'] = '> 12 YEARS'

    # Nb années avant la demande où le client a changé son numéro de téléphone
    app_test['YEARS_LAST_PHONE_CHANGE'] = round(app_test['DAYS_LAST_PHONE_CHANGE'] / 365 * -1, 2)
    del app_test['DAYS_LAST_PHONE_CHANGE']
    app_test.loc[app_test['YEARS_LAST_PHONE_CHANGE'] <= 1, 'YEARS_LAST_PHONE_CHANGE_RANGE'] = '< 1 YEAR'
    app_test.loc[(app_test['YEARS_LAST_PHONE_CHANGE'] > 1) &
                 (app_test['YEARS_LAST_PHONE_CHANGE'] <= 2), 'YEARS_LAST_PHONE_CHANGE_RANGE'] = 'BETWEEN 1 & 2 YEARS'
    app_test.loc[(app_test['YEARS_LAST_PHONE_CHANGE'] > 2) &
                 (app_test['YEARS_LAST_PHONE_CHANGE'] <= 4), 'YEARS_LAST_PHONE_CHANGE_RANGE'] = 'BETWEEN 2 & 4 YEARS'
    app_test.loc[app_test['YEARS_LAST_PHONE_CHANGE'] > 4, 'YEARS_LAST_PHONE_CHANGE_RANGE'] = '> 4 YEARS'

    # Traitements des variables qualitatives

    # Genre
    app_test = app_test[app_test['CODE_GENDER'].isin(["F", "M"])]

    # Accompagnateur
    dico_name_type_suite = {'Unaccompanied': 'Unaccompanied',
                            'Family': 'Family',
                            'Spouse, partner': 'Family',
                            'Children': 'Family',
                            'Other_B': 'Other',
                            'Other_A': 'Other',
                            'Group of people': 'Other'}
    app_test['NAME_TYPE_SUITE'] = app_test['NAME_TYPE_SUITE'].map(dico_name_type_suite)

    # Situation professionnelle
    dico_name_income_type = {'Working': 'Private worker',
                             'Commercial associate': 'Commercial associate',
                             'Pensioner': 'Pensioner/Student/Unemployed',
                             'State servant': 'Public worker',
                             'Unemployed': 'Pensioner/Student/Unemployed',
                             'Student': 'Pensioner/Student/Unemployed',
                             'Businessman': 'Other',
                             'Maternity leave': 'Pensioner/Student/Unemployed'}
    app_test['NAME_INCOME_TYPE'] = app_test['NAME_INCOME_TYPE'].map(dico_name_income_type)

    # Education
    dico_name_education_type = {'Secondary / secondary special': 'Lower Secondary & Secondary',
                                'Higher education': 'Higher education',
                                'Incomplete higher': 'Incomplete higher',
                                'Lower secondary': 'Lower Secondary & Secondary',
                                'Academic degree': 'Higher education', }
    app_test['NAME_EDUCATION_TYPE'] = app_test['NAME_EDUCATION_TYPE'].map(dico_name_education_type)
    app_test = app_test[~app_test['NAME_FAMILY_STATUS'].isin(["Unknown"])]

    # Situation familiale
    dico_name_family_status = {'Married': 'Married',
                               'Single / not married': 'Not Married',
                               'Civil marriage': 'Married',
                               'Separated': 'Separated',
                               'Widow': 'Widow', }
    app_test['NAME_FAMILY_STATUS'] = app_test['NAME_FAMILY_STATUS'].map(dico_name_family_status)

    # Logement
    dico_name_housing_type = {'House / apartment': 'Owner',
                              'With parents': 'No or low rent',
                              'Municipal apartment': 'No or low rent',
                              'Rented apartment': 'Tenant',
                              'Office apartment': 'No or low rent',
                              'Co-op apartment': 'Tenant'}
    app_test['NAME_HOUSING_TYPE'] = app_test['NAME_HOUSING_TYPE'].map(dico_name_housing_type)

    # Moment où le client a fait la demande
    dico_weekday_process = {'TUESDAY': 'working days',
                            'WEDNESDAY': 'working days',
                            'MONDAY': 'working days',
                            'THURSDAY': 'working days',
                            'FRIDAY': 'working days',
                            'SATURDAY': 'week end',
                            'SUNDAY': 'week end'}
    app_test['WEEKDAY_APPR_PROCESS_START'] = app_test['WEEKDAY_APPR_PROCESS_START'].map(dico_weekday_process)

    # Secteur d'activité
    dico_orga_type = {'Industry: type 6': 'Industry',
                      'Industry: type 13': 'Industry',
                      'Industry: type 8': 'Industry',
                      'Industry: type 5': 'Industry',
                      'Industry: type 10': 'Industry',
                      'Industry: type 12': 'Industry',
                      'Industry: type 2': 'Industry',
                      'Industry: type 9': 'Industry',
                      'Industry: type 3': 'Industry',
                      'Industry: type 7': 'Industry',
                      'Industry: type 1': 'Industry',
                      'Industry: type 4': 'Industry',
                      'Industry: type 11': 'Industry',
                      'Business Entity Type 3': 'Business Entity',
                      'Business Entity Type 2': 'Business Entity',
                      'Business Entity Type 1': 'Business Entity',
                      'Transport: type 1': 'Transport',
                      'Transport: type 2': 'Transport',
                      'Transport: type 3': 'Transport',
                      'Transport: type 4': 'Transport',
                      'Trade: type 7': 'Trade',
                      'Trade: type 4': 'Trade',
                      'Trade: type 5': 'Trade',
                      'Trade: type 1': 'Trade',
                      'Trade: type 6': 'Trade',
                      'Trade: type 3': 'Trade',
                      'Trade: type 2': 'Trade',
                      'Bank': 'Bank/Insurance',
                      'Insurance': 'Bank/Insurance',
                      'School': 'School/University/Kindergarten',
                      'University': 'School/University/Kindergarten',
                      'Kindergarten': 'School/University/Kindergarten',
                      'Government': 'Government/Military/Security Ministries',
                      'Military': 'Government/Military/Security Ministries',
                      'Security Ministries': 'Government/Military/Security Ministries',
                      'Telecom': 'Telecom',
                      'Mobile': 'Telecom',
                      'Electricity': 'Electricity/Construction',
                      'Construction': 'Electricity/Construction',
                      'Services': 'Services',
                      'Advertising': 'Services',
                      'Legal Services': 'Services',
                      'Hotel': 'Hotel/Restaurant',
                      'Restaurant': 'Hotel/Restaurant',
                      'Cleaning': 'Cleaning/Housing',
                      'Housing': 'Cleaning/Housing',
                      'Medicine': 'Medical',
                      'Emergency': 'Medical',
                      'Security': 'Police/Security',
                      'Police': 'Police/Security',
                      'Religion': 'Other',
                      'Other': 'Other',
                      'XNA': 'Other',
                      'Self-employed': 'Self-employed',
                      'Postal': 'Postal',
                      'Agriculture': 'Agriculture',
                      'Culture': 'Culture',
                      'Realtor': 'Realtor', }
    app_test['ORGANIZATION_TYPE'] = app_test['ORGANIZATION_TYPE'].map(dico_orga_type)

    # Variables à supprimer
    var_quanti_a_suppr = ['AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_DAY',
                          'AMT_REQ_CREDIT_BUREAU_MON', 'AMT_REQ_CREDIT_BUREAU_QRT',
                          'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE']
    var_quali_a_suppr = ['FLAG_MOBIL', 'FLAG_CONT_MOBILE', 'FLAG_EMAIL', 'WEEKDAY_APPR_PROCESS_START',
                         'LIVE_REGION_NOT_WORK_REGION', 'FLAG_DOCUMENT_4', 'FLAG_DOCUMENT_5', 'FLAG_DOCUMENT_7',
                         'FLAG_DOCUMENT_10', 'FLAG_DOCUMENT_12', 'FLAG_DOCUMENT_17', 'FLAG_DOCUMENT_19',
                         'FLAG_DOCUMENT_20', 'FLAG_DOCUMENT_21']
    var_suppr = var_quanti_a_suppr + var_quali_a_suppr
    app_test = app_test.drop(var_suppr, axis=1)

    # Feature engineering

    app_test['CREDIT_DURATION'] = app_test['AMT_CREDIT'] / app_test['AMT_ANNUITY']
    app_test['DEBT_RATIO'] = app_test['AMT_ANNUITY'] / app_test['AMT_INCOME_TOTAL']
    app_test['PAYMENT_RATE'] = app_test['AMT_ANNUITY'] / app_test['AMT_CREDIT']
    app_test['INCOME_PER_PERSON'] = app_test['AMT_INCOME_TOTAL'] / app_test['CNT_FAM_MEMBERS']
    app_test['INCOME_CREDIT_PERC'] = app_test['AMT_INCOME_TOTAL'] / app_test['AMT_CREDIT']
    app_test['CREDIT_GOODS_PERC'] = app_test['AMT_CREDIT'] / app_test['AMT_GOODS_PRICE']

    # Durée du crédit
    app_test.loc[app_test['CREDIT_DURATION'] < 15, 'CREDIT_DURATION_RANGE'] = 'SHORT < 15 Years'
    app_test.loc[(app_test['CREDIT_DURATION'] >= 15) &
                 (app_test['CREDIT_DURATION'] <= 20), 'CREDIT_DURATION_RANGE'] = 'MEDIUM BETWEEN 15 & 20 Years'
    app_test.loc[app_test['CREDIT_DURATION'] > 20, 'CREDIT_DURATION_RANGE'] = 'LONG > 20 Years'

    # Taux d'endettement
    app_test.loc[app_test['DEBT_RATIO'] < 0.15, 'DEBT_RATIO_RANGE'] = 'VERY LOW < 15'
    app_test.loc[(app_test['DEBT_RATIO'] >= 0.15) &
                 (app_test['DEBT_RATIO'] <= 0.20), 'DEBT_RATIO_RANGE'] = 'LOW BETWEEN 15 & 20 Years'
    app_test.loc[(app_test['DEBT_RATIO'] > 0.20) &
                 (app_test['DEBT_RATIO'] <= 0.35), 'DEBT_RATIO_RANGE'] = 'MEDIUM BETWEEN 21 & 35'
    app_test.loc[app_test['DEBT_RATIO'] > 0.35, 'DEBT_RATIO_RANGE'] = 'VERY HIGH > 35%'

    # Revenus par personne
    app_test.loc[app_test['INCOME_PER_PERSON'] < 45000, 'INCOME_PER_PERSON_RANGE'] = 'LOW < 45K'
    app_test.loc[(app_test['INCOME_PER_PERSON'] >= 45000) &
                 (app_test['INCOME_PER_PERSON'] <= 75000), 'INCOME_PER_PERSON_RANGE'] = 'MEDIUM BETWEEN 45K & 75K'
    app_test.loc[app_test['INCOME_PER_PERSON'] > 75000, 'INCOME_PER_PERSON_RANGE'] = 'HIGH > 75K'
    app_test = app_test.set_index('SK_ID_CURR')

    del var_quanti_a_suppr, var_quali_a_suppr, var_suppr
    gc.collect()
    return app_test


def preprocess_bureau_balance():
    """Fonction de preprocessing des datasets bureau et bureau_balance :
    merge, traitement des données manquantes, traitement des outliers et des modalités,
    feature engineering

    Arguments:
    --------------------------------


    return:
    --------------------------------
    Dataframe preprocessé"""

    # Chargement des datasets

    # Get the absolute path to the data directory
    script_path = Path(__file__).resolve()
    data_path_bureau = script_path.parent / 'data' / 'bureau.csv'
    bureau = pd.read_csv(data_path_bureau, sep=',', encoding='utf-8')
    data_path_bureau_balance = script_path.parent / 'data' / 'bureau_balance.csv'
    bureau_balance = pd.read_csv(data_path_bureau_balance, sep=',', encoding='utf-8')

    # Preprocessing bureau_balance

    dico_status = {'C': 'Closed',
                   '0': 'No DPD',
                   'X': 'Unknown',
                   '1': 'Max 30 DPD',
                   '2': 'Between 31 and 120 DPD',
                   '3': 'Between 31 and 120 DPD',
                   '4': 'Between 31 and 120 DPD',
                   '5': 'More than 120 DPD or written off'}
    bureau_balance['STATUS'] = bureau_balance['STATUS'].map(dico_status)
    bureau_balance, bureau_balance_cat = categories_encoder(bureau_balance, nan_as_category=False)

    # Agrégations

    bureau_balance_aggreg = {'MONTHS_BALANCE': ['size']}
    for col in bureau_balance_cat:
        bureau_balance_aggreg[col] = ['sum']

    # Groupe par SK_ID_BUREAU avec application du dictionnaire des agrégations
    bureau_balance = bureau_balance.groupby('SK_ID_BUREAU').agg(bureau_balance_aggreg)

    # Renommage des colonnes pour éviter les niveaux multiples
    bureau_balance.columns = pd.Index([e[0] + "_" + e[1].upper() for e in bureau_balance.columns.tolist()])

    # Fusion de bureau et bureau balance et suppression de la variable SK_ID_BUREAU
    bureau_bb = bureau.join(bureau_balance, how='left', on='SK_ID_BUREAU')
    bureau_bb.drop(['SK_ID_BUREAU'], axis=1, inplace=True)

    # Traitement des données manquantes

    # 0 pour les variables STATUS et MONTHS_BALANCE_SIZE
    for col in bureau_bb.columns[bureau_bb.columns.str.contains('STATUS')]:
        bureau_bb[col] = bureau_bb[col].fillna(0)
    bureau_bb['MONTHS_BALANCE_SIZE'] = bureau_bb['MONTHS_BALANCE_SIZE'].fillna(0)

    # Suppression des variables avec 30% ou plus de NaN
    bureau_bb = tx_rempl_min(bureau_bb, 70)

    # Médiane pour variables avec taux de remplissage > 70%
    df_nan = nan_a_retraiter(bureau_bb)
    col_med = df_nan[(df_nan['Tx_rempl'] > 70) &
                     (df_nan['dtypes'] == 'float64')].index.tolist()
    for col in col_med:
        bureau_bb[col] = bureau_bb[col].fillna(bureau_bb[col].median())

    # Suppression de variables
    bureau_bb.drop(['CREDIT_DAY_OVERDUE', 'CNT_CREDIT_PROLONG',
                    'AMT_CREDIT_SUM_OVERDUE'], axis=1, inplace=True)

    # Feature engineering

    # Nb années avant la demande où le client a demandé un crédit auprès de Crédit Bureau
    bureau_bb['YEARS_CREDIT'] = round(bureau_bb['DAYS_CREDIT'] / 365 * -1, 2)
    del bureau_bb['DAYS_CREDIT']

    # Durée restante du crédit souscrit auprès de Credit Bureau en année au moment
    # de l'application chez Home Credit
    bureau_bb['YEARS_CREDIT_ENDDATE'] = round(bureau_bb['DAYS_CREDIT_ENDDATE'] / 365 * -1, 2)
    del bureau_bb['DAYS_CREDIT_ENDDATE']
    del bureau_bb['DAYS_CREDIT_UPDATE']

    # Suppression de la variable CREDIT_CURRENCY
    del bureau_bb['CREDIT_CURRENCY']

    dico_credit_active = {'Closed': 'Closed',
                          'Active': 'Active',
                          'Sold': 'Sold / Bad Debt',
                          'Bad debt': 'Sold / Bad Debt'}

    # Statuts des crédits déclarés par Bureau Crédit
    bureau_bb['CREDIT_ACTIVE'] = bureau_bb['CREDIT_ACTIVE'].map(dico_credit_active)

    # Types de crédit
    dico_credit_type = {'Consumer credit': 'Consumer credit',
                        'Credit card': 'Credit card',
                        'Car loan': 'Car loan',
                        'Mortgage': 'Mortgage',
                        'Microloan': 'Microloan',
                        'Loan for business development': 'Other',
                        'Another type of loan': 'Other',
                        'Unknown type of loan': 'Other',
                        'Loan for working capital replenishment': 'Other',
                        'Cash loan (non-earmarked)': 'Other',
                        'Real estate loan': 'Other',
                        'Loan for the purchase of equipment': 'Other',
                        'Loan for purchase of shares (margin lending)': 'Other',
                        'Mobile operator loan': 'Other',
                        'Interbank credit': 'Other', }
    bureau_bb['CREDIT_TYPE'] = bureau_bb['CREDIT_TYPE'].map(dico_credit_type)

    # Suppression de variables
    del bureau_bb['YEARS_CREDIT']
    del bureau_bb['STATUS_No DPD_SUM']
    del bureau_bb['STATUS_Unknown_SUM']
    del bureau_bb['MONTHS_BALANCE_SIZE']

    # Preprocessing variables catégorielles
    bureau_bb_feat, bureau_bb_feat_cat = categories_encoder(bureau_bb, nan_as_category=False)

    # Actualisation de la liste des variables catégorielles de bureau_balance
    bureau_balance_cat.remove('STATUS_No DPD')
    bureau_balance_cat.remove('STATUS_Unknown')

    # Dictionnaire des agrégations
    num_aggregations = {
        'AMT_CREDIT_SUM': ['sum', 'mean'],
        'AMT_CREDIT_SUM_DEBT': ['sum', 'mean'],
        'YEARS_CREDIT_ENDDATE': ['mean', 'max']
    }
    cat_aggregations = {}
    for cat in bureau_bb_feat_cat:
        cat_aggregations[cat] = ['sum']
    for cat in bureau_balance_cat:
        cat_aggregations[cat + "_SUM"] = ['sum']

    # Groupe par SK_ID_CURR avec application du dictionnaire des agrégations
    bureau_bb_feat_gby = bureau_bb_feat.groupby('SK_ID_CURR').agg({**num_aggregations, **cat_aggregations})

    # Renommage des colonnes pour éviter les niveaux multiples
    bureau_bb_feat_gby.columns = pd.Index(['BURO_' + e[0] + "_" + e[1].upper()
                                           for e in bureau_bb_feat_gby.columns.tolist()])

    del bureau, bureau_balance, bureau_bb, bureau_bb_feat
    gc.collect()

    return bureau_bb_feat_gby


def preprocess_previous_app():
    """Fonction de preprocessing du dataset previous application :
    traitement des données manquantes, feature engineering et
    sélection de variables

    Arguments:
    --------------------------------


    return:
    --------------------------------
    Dataframe preprocessé"""

    # Chargement du dataset

    # Get the absolute path to the data directory
    script_path = Path(__file__).resolve()
    data_path = script_path.parent / 'data' / 'previous_application.csv'
    previous_app = pd.read_csv(data_path, sep=',', encoding='utf-8')

    # Traitement des données manquantes

    # Suppression des variables avec 30% ou plus de NaN
    previous_app = tx_rempl_min(previous_app, 70)

    # Iterative imputer pour les variables corrélées entre-elles
    previous_app['AMT_CREDIT'] = iterative_imputer_function(
        previous_app[['AMT_CREDIT', 'AMT_APPLICATION']])[:, 0]
    previous_app['AMT_ANNUITY'] = iterative_imputer_function(
        previous_app[['AMT_ANNUITY', 'AMT_CREDIT', 'AMT_GOODS_PRICE',
                      'AMT_APPLICATION', 'CNT_PAYMENT']])[:, 0]

    # Médiane et mode pour variables avec taux de NaN > 70% (hors variable TARGET)
    df_nan = nan_a_retraiter(previous_app)
    col_med = df_nan[(df_nan['Tx_rempl'] > 70) &
                     (df_nan['dtypes'] == 'float64')].index.tolist()
    for col in col_med:
        previous_app[col] = previous_app[col].fillna(previous_app[col].median())

    # Imputation de la variable NAME_TYPE_SUITE par le mode
    previous_app['PRODUCT_COMBINATION'] = previous_app['PRODUCT_COMBINATION'].fillna(
        previous_app['PRODUCT_COMBINATION'].mode()[0])

    # Feature engineering

    previous_app_feat = previous_app.copy()
    del previous_app_feat['SELLERPLACE_AREA']
    del previous_app_feat['NFLAG_LAST_APPL_IN_DAY']

    # Nb d'années depuis la demande précédente
    previous_app_feat['YEARS_DECISION'] = round(previous_app_feat['DAYS_DECISION'] / 365 * -1, 2)
    del previous_app_feat['DAYS_DECISION']

    # Création des ratios
    previous_app_feat['CREDIT_DURATION'] = np.where(previous_app_feat['AMT_ANNUITY'] != 0,
                                                    previous_app_feat['AMT_CREDIT'] / previous_app_feat['AMT_ANNUITY'],
                                                    0)
    previous_app_feat['CREDIT_GOODS_PERC'] = np.where(previous_app_feat['AMT_GOODS_PRICE'] != 0,
                                                      previous_app_feat['AMT_CREDIT'] / previous_app_feat[
                                                          'AMT_GOODS_PRICE'],
                                                      0)
    previous_app_feat['PAYMENT_RATE'] = np.where(previous_app_feat['AMT_CREDIT'] != 0,
                                                 previous_app_feat['AMT_ANNUITY'] / previous_app_feat['AMT_CREDIT'],
                                                 0)
    # Dictionnaire des agrégations pour les variables numériques
    num_aggregations = {
        'SK_ID_PREV': ['count'],
        'YEARS_DECISION': ['mean'],
        'CREDIT_DURATION': ['mean'],
        'CREDIT_GOODS_PERC': ['mean'],
        'PAYMENT_RATE': ['mean'],
    }

    # Aggregations variables catégorielles
    previous_app_feat, previous_app_cat = categories_encoder(previous_app_feat, nan_as_category=False)
    cat_aggregations = {}
    for cat in previous_app_cat:
        cat_aggregations[cat] = ['mean']

    # Groupe par SK_ID_CURR + agrégation
    previous_app_feat_gby = previous_app_feat.groupby('SK_ID_CURR').agg({**num_aggregations,
                                                                         **cat_aggregations})

    # On renomme les colonnes pour supprimer le multiindex
    previous_app_feat_gby.columns = pd.Index(['PREV_' + e[0] + "_" + e[1].upper()
                                              for e in previous_app_feat_gby.columns.tolist()])

    # On renomme SK_ID_PREV_COUNT en PREV_NB_CREDIT et YEARS_DECISION_MEAN en PREV_YEARS_DECISION_MEAN
    previous_app_feat_gby = previous_app_feat_gby.rename(columns={'PREV_SK_ID_PREV_COUNT': 'PREV_NB_CREDIT'})

    # Indicateurs agrégés crédits approuvés
    cred_approuv = previous_app_feat[previous_app_feat['NAME_CONTRACT_STATUS_Approved'] == 1]
    cred_approuv_agg = cred_approuv.groupby('SK_ID_CURR').agg(num_aggregations)
    cred_approuv_agg.columns = pd.Index(
        ['APPROVED_' + e[0] + "_" + e[1].upper() for e in cred_approuv_agg.columns.tolist()])
    previous_app_agg = previous_app_feat_gby.join(cred_approuv_agg, how='left', on='SK_ID_CURR')

    # Indicateurs agrégés crédits refusés
    cred_refus = previous_app_feat[previous_app_feat['NAME_CONTRACT_STATUS_Refused'] == 1]
    cred_refus_agg = cred_refus.groupby('SK_ID_CURR').agg(num_aggregations)
    cred_refus_agg.columns = pd.Index(['REFUSED' + e[0] + "_" + e[1].upper() for e in cred_refus_agg.columns.tolist()])
    previous_app_agg = previous_app_agg.join(cred_refus_agg, how='left', on='SK_ID_CURR')

    # Imputation des valeurs manquantes par la constante 0
    for col in cred_approuv_agg.columns:
        previous_app_agg[col] = previous_app_agg[col].fillna(0)
    for col in cred_refus_agg.columns:
        previous_app_agg[col] = previous_app_agg[col].fillna(0)

    del previous_app, previous_app_feat, previous_app_feat_gby, cred_approuv_agg, cred_refus_agg
    gc.collect()

    return previous_app_agg


def preprocess_POS_CASH_balance():
    """Fonction de preprocessing du dataset POS_CASH_balance :
    traitement des données manquantes, feature engineering et
    sélection de variables

    Arguments:
    --------------------------------


    return:
    --------------------------------
    Dataframe preprocessé"""

    # Chargement du dataset

    # Get the absolute path to the data directory
    script_path = Path(__file__).resolve()
    data_path = script_path.parent / 'data' / 'POS_CASH_balance.csv'
    pos_cash = pd.read_csv(data_path, sep=',', encoding='utf-8')

    # Traitement des données manquantes (médiane)

    for col in ['CNT_INSTALMENT', 'CNT_INSTALMENT_FUTURE']:
        pos_cash[col] = pos_cash[col].fillna(pos_cash[col].median())

    # Feature engineering

    del pos_cash['SK_DPD']
    del pos_cash['SK_DPD_DEF']
    del pos_cash['NAME_CONTRACT_STATUS']
    del pos_cash['CNT_INSTALMENT']
    pos_cash_feat, pos_cash_cat = categories_encoder(pos_cash, nan_as_category=False)

    # Dictionnaire des agrégations pour les variables numériques
    aggregations = {
        'MONTHS_BALANCE': ['mean'],
        'CNT_INSTALMENT_FUTURE': ['mean'],
        'SK_ID_PREV': ['count']
    }

    # Groupe par SK_ID_CURR + agrégation
    pos_cash_feat_gby = pos_cash_feat.groupby('SK_ID_CURR').agg(aggregations)

    # On renomme les colonnes pour supprimer le multi-index
    pos_cash_feat_gby.columns = pd.Index(['POS_' + e[0] + "_" + e[1].upper()
                                          for e in pos_cash_feat_gby.columns.tolist()])

    # On renomme POS_SK_ID_PREV_COUNT en POS_NB_CREDIT
    pos_cash_feat_gby = pos_cash_feat_gby.rename(columns={'POS_SK_ID_PREV_COUNT': 'POS_NB_CREDIT'})

    del pos_cash, pos_cash_feat
    gc.collect()

    return pos_cash_feat_gby


def preprocess_installments_payments():
    """Fonction de preprocessing du dataset installments_payments :
    traitement des données manquantes, feature engineering et
    sélection de variables

    Arguments:
    --------------------------------


    return:
    --------------------------------
    Dataframe preprocessé"""

    # Chargement du dataset

    # Get the absolute path to the data directory
    script_path = Path(__file__).resolve()
    data_path = script_path.parent / 'data' / 'installments_payments.csv'
    install_pay = pd.read_csv(data_path, sep=',', encoding='utf-8')

    # Traitement des données manquantes (médiane)

    for col in ['DAYS_ENTRY_PAYMENT', 'AMT_PAYMENT']:
        install_pay[col] = install_pay[col].fillna(install_pay[col].median())

    # Feature engineering

    del install_pay['NUM_INSTALMENT_VERSION']
    install_pay['PAYMENT_DIFF'] = install_pay['AMT_INSTALMENT'] - install_pay['AMT_PAYMENT']
    install_pay['DAYS_PAST_DUE'] = install_pay['DAYS_ENTRY_PAYMENT'] - install_pay['DAYS_INSTALMENT']
    install_pay['DAYS_PAST_DUE'] = install_pay['DAYS_PAST_DUE'].apply(lambda x: x if x > 0 else 0)
    install_pay['DAYS_BEFORE_DUE'] = install_pay['DAYS_INSTALMENT'] - install_pay['DAYS_ENTRY_PAYMENT']
    install_pay['DAYS_BEFORE_DUE'] = install_pay['DAYS_BEFORE_DUE'].apply(lambda x: x if x > 0 else 0)

    # Dictionnaire des agrégations pour les variables numériques
    aggregations = {
        'DAYS_PAST_DUE': ['mean'],
        'DAYS_BEFORE_DUE': ['mean'],
        'PAYMENT_DIFF': ['mean'],
        'SK_ID_PREV': ['count']
    }

    # Groupe par SK_ID_CURR + agrégation
    install_pay_feat_gby = install_pay.groupby('SK_ID_CURR').agg(aggregations)

    # On renomme les colonnes pour supprimer le multi-index
    install_pay_feat_gby.columns = pd.Index(['INSTAL_' + e[0] + "_" + e[1].upper()
                                             for e in install_pay_feat_gby.columns.tolist()])

    # On renomme INSTAL_SK_ID_PREV_COUNT en INSTAL_NB_INSTAL
    install_pay_feat_gby = install_pay_feat_gby.rename(columns={'INSTAL_SK_ID_PREV_COUNT': 'INSTAL_NB_INSTAL'})

    del install_pay
    gc.collect()

    return install_pay_feat_gby


def preprocess_credit_card_balance():
    """Fonction de preprocessing du dataset credit_card_balance :
    traitement des données manquantes, feature engineering et
    sélection de variables

    Arguments:
    --------------------------------


    return:
    --------------------------------
    Dataframe preprocessé"""

    # Chargement du dataset

    # Get the absolute path to the data directory
    script_path = Path(__file__).resolve()
    data_path = script_path.parent / 'data' / 'credit_card_balance.csv'
    cc_balance = pd.read_csv(data_path, sep=',', encoding='utf-8')

    # Traitement des données manquantes (médiane)

    df_nan = nan_a_retraiter(cc_balance)
    col_med = df_nan[(df_nan['Tx_rempl'] >= 80)].index.tolist()
    for col in col_med:
        cc_balance[col] = cc_balance[col].fillna(cc_balance[col].median())

    # Suppression variables suite analyses
    cc_balance.drop(['AMT_DRAWINGS_ATM_CURRENT', 'AMT_DRAWINGS_CURRENT', 'AMT_DRAWINGS_OTHER_CURRENT',
                     'AMT_DRAWINGS_POS_CURRENT', 'CNT_DRAWINGS_ATM_CURRENT', 'CNT_DRAWINGS_CURRENT',
                     'CNT_DRAWINGS_OTHER_CURRENT', 'CNT_DRAWINGS_POS_CURRENT', 'SK_DPD', 'SK_DPD_DEF',
                     'NAME_CONTRACT_STATUS', 'AMT_INST_MIN_REGULARITY', 'AMT_PAYMENT_CURRENT',
                     'AMT_PAYMENT_TOTAL_CURRENT', 'AMT_RECEIVABLE_PRINCIPAL', 'AMT_RECIVABLE',
                     'AMT_RECIVABLE', 'AMT_TOTAL_RECEIVABLE', 'AMT_TOTAL_RECEIVABLE'],
                    axis=1, inplace=True)

    # Feature engineering

    # Groupe par SK_ID_PREV + agrégations
    num_aggregations = {
        'SK_ID_PREV': ['count'],
        'MONTHS_BALANCE': ['mean'],
        'AMT_BALANCE': ['mean'],
        'AMT_CREDIT_LIMIT_ACTUAL': ['mean'],
        'CNT_INSTALMENT_MATURE_CUM': ['mean'],
    }
    cc_balance_feat_gby = cc_balance.groupby('SK_ID_CURR').agg(num_aggregations)

    # On renomme les colonnes pour supprimer le multi-index
    cc_balance_feat_gby.columns = pd.Index(['CC_' + e[0] + "_" + e[1].upper()
                                            for e in cc_balance_feat_gby.columns.tolist()])

    # On renomme CC_SK_ID_PREV_COUNT en PREV_NB_CREDIT
    cc_balance_feat_gby = cc_balance_feat_gby.rename(columns={'CC_SK_ID_PREV_COUNT': 'CC_NB_CREDIT'})

    del cc_balance
    gc.collect()

    return cc_balance_feat_gby


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print(f"{title} fait en {np.round(time.time() - t0, 0)}s ")


# --------------------------------------------------------------------------------
# ----------------------- PREPROCESSING GLOBAL -----------------------------------
# --------------------------------------------------------------------------------

def preprocessing_no_NaN():
    """Fonction de preprocessing des différents datasets (traitement des données manquantes,
    feature engineering et sélection de variables) et jointure avec le dataset principal
    application_train. La fonction exporte également les données au format csv pour
    utilisation dans l'application (gain de temps de traitement).

        Arguments:
        --------------------------------


        return:
        --------------------------------
        Dataframe preprocessé et consolidé"""

    with timer("Processing application_train"):
        df = preprocess_app_train()
        print("Application_train shape:", df.shape)
    with timer("Processing bureau et bureau_balance"):
        bureau = preprocess_bureau_balance()
        print("Bureau shape:", bureau.shape)
        df = df.join(bureau, how='left', on='SK_ID_CURR')
        del bureau
        gc.collect()
    with timer("Processing previous_applications"):
        prev = preprocess_previous_app()
        print("Previous applications shape:", prev.shape)
        df = df.join(prev, how='left', on='SK_ID_CURR')
        del prev
        gc.collect()
    with timer("Processing POS-CASH balance"):
        pos = preprocess_POS_CASH_balance()
        print("Pos-cash balance shape:", pos.shape)
        df = df.join(pos, how='left', on='SK_ID_CURR')
        del pos
        gc.collect()
    with timer("Processing installments payments"):
        ins = preprocess_installments_payments()
        print("Installments payments shape:", ins.shape)
        df = df.join(ins, how='left', on='SK_ID_CURR')
        del ins
        gc.collect()
    with timer("Processing credit card balance"):
        cc = preprocess_credit_card_balance()
        print("Credit card balance shape:", cc.shape)
        df = df.join(cc, how='left', on='SK_ID_CURR')
        df = tx_rempl_min(df, 80)
        df_nan = nan_a_retraiter(df)
        col_med = df_nan[(df_nan['Tx_rempl'] > 80) &
                         (df_nan['dtypes'] == 'float64') &
                         (~ df_nan.index.str.contains('TARGET'))].index.tolist()
        for col in col_med:
            df[col] = df[col].fillna(df[col].median())
        del cc, df_nan, col_med
        gc.collect()
        df.index = df.index.map(str)
        df.to_csv('df.csv')
    return df


# --------------------------------------------------------------------------------
# ------------ PREPROCESSING GLOBAL APP_TEST POUR DATA DRIFT----------------------
# --------------------------------------------------------------------------------

def preprocessing_no_NaN_app_test():
    """Fonction de preprocessing des différents datasets (traitement des données manquantes,
    feature engineering et sélection de variables) et jointure avec le dataset
    application_test pour analyse du data drift.

        Arguments:
        --------------------------------


        return:
        --------------------------------
        Dataframe preprocessé et consolidé"""

    with timer("Processing application_test"):
        df = preprocess_app_test()
        print("Application_test shape:", df.shape)
    with timer("Processing bureau et bureau_balance"):
        bureau = preprocess_bureau_balance()
        print("Bureau shape:", bureau.shape)
        df = df.join(bureau, how='left', on='SK_ID_CURR')
        del bureau
        gc.collect()
    with timer("Processing previous_applications"):
        prev = preprocess_previous_app()
        print("Previous applications shape:", prev.shape)
        df = df.join(prev, how='left', on='SK_ID_CURR')
        del prev
        gc.collect()
    with timer("Processing POS-CASH balance"):
        pos = preprocess_POS_CASH_balance()
        print("Pos-cash balance shape:", pos.shape)
        df = df.join(pos, how='left', on='SK_ID_CURR')
        del pos
        gc.collect()
    with timer("Processing installments payments"):
        ins = preprocess_installments_payments()
        print("Installments payments shape:", ins.shape)
        df = df.join(ins, how='left', on='SK_ID_CURR')
        del ins
        gc.collect()
    with timer("Processing credit card balance"):
        cc = preprocess_credit_card_balance()
        print("Credit card balance shape:", cc.shape)
        df = df.join(cc, how='left', on='SK_ID_CURR')
        df = tx_rempl_min(df, 80)
        df_nan = nan_a_retraiter(df)
        col_med = df_nan[(df_nan['Tx_rempl'] > 80) &
                         (df_nan['dtypes'] == 'float64') &
                         (~ df_nan.index.str.contains('TARGET'))].index.tolist()
        for col in col_med:
            df[col] = df[col].fillna(df[col].median())
        del cc, df_nan, col_med
        gc.collect()
        df.index = df.index.map(str)
    return df


if __name__ == '__main__':
    preprocessing_no_NaN()
