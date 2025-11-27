# Fonctions pour l'analyse exploratoire

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
import matplotlib.patches as mpatches
import seaborn as sns
from scipy.stats import chi2_contingency  # Chi2
import pingouin as pg  # Test Chi2 d'indépendance
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


# --------------------------------------------------------------------
# ------------------------ SHAPE & NAN -------------------------------
# --------------------------------------------------------------------

def shape_total_nan(df):
    """Fonction qui retourne le nombre de lignes,
    de variables, le nombre total de valeurs manquantes et
    le pourcentage associé

    Arguments:
    --------------------------------
    df: dataframe: tableau en entrée, obligatoire


    return:
    --------------------------------
    None"""

    missing = df.isna().sum().sum()
    missing_percent = round(missing
                            / (df.shape[0] * df.shape[1])
                            * 100,
                            2)

    print(f"Nombre de lignes: {df.shape[0]}")
    print(f"Nombre de colonnes: {df.shape[1]}")
    print(f"Nombre total de NaN du dataset: {missing}")
    print(f"% total de NaN du dataset: {missing_percent}%")


# --------------------------------------------------------------------
# ---------------- DESCRIPTION DES VARIABLES -------------------------
# --------------------------------------------------------------------

def describe_variables_light(data):
    """ Fonction qui prend un dataframe en entrée, et retourne un
    récapitulatif qui contient le nom des variables, leur type, un
    exemple de modalité, le nombre total de lignes, le nombre et
    pourcentage de valeurs distinctes, le nombre et pourcentage de
    valeurs non manquantes et de valeurs manquantes (NaN).

    Arguments:
    --------------------------------
    data: dataframe: tableau en entrée, obligatoire


    return:
    --------------------------------
    dataframe qui décrit les variables"""

    # Choix du nom des variables à afficher
    df = pd.DataFrame(columns=[
        'Variable name', 'Variable type', 'Example', 'Rows', 'Distinct',
        '% distinct', 'Not NaN', '% Not NaN', 'NaN', '% NaN'])

    # Pour chaque colonne du dataframe
    for col in data.columns:
        # Définition des variables
        # type de la variable (object, float, int...)
        var_type = data[col].dtypes
        # premier élément notNA
        example = data[data[col].notna()][col].iloc[0]
        # nombre total de lignes
        nb_raw = len(data[col])
        # nombre de valeurs non manquantes
        count = len(data[col]) - data[col].isna().sum()
        # % de valeurs non manquantes
        percent_count = round(data[col].notnull().mean(), 4) * 100
        # nombre de modalités que peut prendre la variable
        distinct = data[col].nunique()
        # % de valeurs distinctes
        percent_distinct = round(data[col].nunique() / len(data[col]), 4)
        percent_distinct = percent_distinct * 100
        # nombre de valeurs manquantes
        missing = data[col].isna().sum()
        # % de valeurs manquantes
        percent_missing = round(data[col].isna().mean(), 4) * 100

        df = pd.concat([df, pd.DataFrame([[col, var_type, example, nb_raw,
                                           distinct, percent_distinct,
                                           count,
                                           percent_count,
                                           missing,
                                           percent_missing]],
                                         columns=['Variable name',
                                                  'Variable type',
                                                  'Example',
                                                  'Rows',
                                                  'Distinct',
                                                  '% distinct',
                                                  'Not NaN',
                                                  '% Not NaN',
                                                  'NaN',
                                                  '% NaN'])])

    return df.reset_index(drop=True)


def describe_variables(data):
    """ Fonction qui prend un dataframe en entrée, et retourne un
    récapitulatif qui contient le nom des variables, leur type, un
    exemple de modalité, le nombre total de lignes, le nombre et
    pourcentage de valeurs distinctes, le nombre et pourcentage de
    valeurs non manquantes et de valeurs manquantes (NaN) et les
    principales statistiques pour les variables numériques (moyenne,
    médiane, distribution, variance, écart type, minimum, quartiles et
    maximum)

    Arguments:
    --------------------------------
    data: dataframe: tableau en entrée, obligatoire


    return:
    --------------------------------
    dataframe qui décrit les variables"""

    # Choix du nom des variables à afficher
    df = pd.DataFrame(columns=[
        'Variable name', 'Variable type', 'Example', 'Rows', 'Distinct',
        '% distinct', 'Not NaN', '% Not NaN', 'NaN', '% NaN', 'Mean',
        'Median', 'Skew', 'Kurtosis', 'Variance', 'Std', 'Min', '25%',
        '75%', 'Max'
    ])

    # Pour chaque colonne du dataframe
    for col in data.columns:

        # Définition des variables
        # type de la variable (object, float, int...)
        var_type = data[col].dtypes
        # premier élément notNA
        example = data[data[col].notna()][col].iloc[0]
        # nombre total de lignes
        nb_raw = len(data[col])
        # nombre de valeurs non manquantes
        count = len(data[col]) - data[col].isna().sum()
        # % de valeurs non manquantes
        percent_count = round(data[col].notnull().mean(), 4) * 100
        # nombre de modalités que peut prendre la variable
        distinct = data[col].nunique()
        # % de valeurs distinctes
        percent_distinct = round(data[col].nunique() / len(data[col]), 4)
        percent_distinct = percent_distinct * 100
        # nombre de valeurs manquantes
        missing = data[col].isna().sum()
        # % de valeurs manquantes
        percent_missing = round(data[col].isna().mean(), 4) * 100

        # Pour les var de type 'int' ou 'float' : on remplit toutes les colonnes
        if var_type == 'int32' or var_type == 'int64' or var_type == 'float':
            df = pd.concat([df, pd.DataFrame([[col, var_type, example, nb_raw,
                                               distinct, percent_distinct,
                                               count,
                                               percent_count,
                                               missing,
                                               percent_missing,
                                               round(data[col].mean(), 2),
                                               round(data[col].median(), 2),
                                               round(data[col].skew(), 2),
                                               round(data[col].kurtosis(), 2),
                                               round(data[col].var(), 2),
                                               round(data[col].std(), 2),
                                               round(data[col].min(), 2),
                                               round(data[col].quantile(0.25),
                                                     2),
                                               round(data[col].quantile(0.75),
                                                     2),
                                               data[col].max()]],
                                             columns=['Variable name',
                                                      'Variable type',
                                                      'Example',
                                                      'Rows',
                                                      'Distinct',
                                                      '% distinct',
                                                      'Not NaN',
                                                      '% Not NaN',
                                                      'NaN',
                                                      '% NaN',
                                                      'Mean',
                                                      'Median',
                                                      'Skew',
                                                      'Kurtosis',
                                                      'Variance',
                                                      'Std',
                                                      'Min',
                                                      '25%',
                                                      '75%',
                                                      'Max'])])

            # Pour les variables d'un autre type : on ne remplit que
            # les variables de compte

        else:
            df = pd.concat([df, pd.DataFrame([[col, var_type, example,
                                               nb_raw, distinct,
                                               percent_distinct,
                                               count,
                                               percent_count, missing,
                                               percent_missing,
                                               '', '', '', '', '', '',
                                               '', '', '', '']],
                                             columns=['Variable name',
                                                      'Variable type',
                                                      'Example',
                                                      'Rows',
                                                      'Distinct',
                                                      '% distinct',
                                                      'Not NaN',
                                                      '% Not NaN',
                                                      'NaN',
                                                      '% NaN',
                                                      'Mean',
                                                      'Median',
                                                      'Skew',
                                                      'Kurtosis',
                                                      'Variance',
                                                      'Std',
                                                      'Min',
                                                      '25%',
                                                      '75%',
                                                      'Max'])])

    return df.reset_index(drop=True)


# --------------------------------------------------------------------
# ----------- DESCRIPTION ET STATISTIQUES ----------------------------
# --------------------------------------------------------------------

def describe_stat(df, fig):
    """Fonction qui prend un dataframe en entrée et
    retourne ses principales statistiques descriptives.

    Arguments:
    --------------------------------
    df: dataframe: tableau en entrée, obligatoire
    fig : taille de la figure

    return:
    --------------------------------
    dataframe qui décrit les variables"""

    print('-------------------------------------------------')
    print('Visualisation des 5 premières lignes du dataset')
    print('-------------------------------------------------')
    display(df.head())

    print('-------------------------------------------------')
    print('Visualisation des 5 dernières lignes du dataset')
    print('-------------------------------------------------')
    display(df.tail())

    print('-------------------------------------------------')
    print('Dimensions du dataset')
    print('-------------------------------------------------')
    print()
    shape_total_nan(df)
    print()

    # print('-------------------------------------------------')
    # print('Heatmap des données manquantes')
    # print('-------------------------------------------------')
    # plt.figure(figsize=(15,7))
    # plt.title('Complétion par colonne', fontweight = 'bold', fontsize = 12)
    # sns.heatmap(df.isna(), yticklabels = False, cbar = False)
    # plt.show()

    print('-------------------------------------------------')
    print('Taux de remplissage')
    print('-------------------------------------------------')
    not_nan_percent = round(df.notna().mean().sort_values(ascending=False), 4) * 100
    # plt.figure(figsize=(12, 30))
    ax = sns.barplot(y=not_nan_percent.index, x=not_nan_percent.values, palette='Purples_r')
    plt.title("Taux de remplissage des variables")
    plt.ylabel("")
    plt.xlabel("Taux de remplissage")
    etiquette_h(ax)
    plt.show()

    print('-------------------------------------------------')
    print('Principales statistiques du jeu de données')
    print('-------------------------------------------------')
    df_desc = describe_variables_light(df)
    display(df_desc)

    print('-------------------------------------------------')
    print('Lignes dupliquées')
    print('-------------------------------------------------')
    print()
    print(f"Lignes en doublons: {df.duplicated().sum()}")

    return df_desc


# --------------------------------------------------------------------
# ---------------- FONCTIONS VALEURS MANQUANTES ----------------------
# --------------------------------------------------------------------

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
    df_concat = pd.concat([not_nan_percent, nan_nb, dtype], keys=['Tx_rempl',
                                                                  'nb_NaN',
                                                                  'dtypes'], axis=1)
    df_concat = df_concat[df_concat['nb_NaN'] > 0]
    # display(df_concat)

    return df_concat


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

    # nan_a_retraiter(df_clean)

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
    array avec les données manquantes estimées par l itérative
    imputer"""

    imp_mean = IterativeImputer(random_state=42)
    X = np.array(df_var_corr)
    imp_mean.fit(X)
    result = imp_mean.transform(X)

    return result


def distrib_imput_nan_quanti(df_before, var, df_after, bins=100):
    """Fonction qui affiche la distribution d'une variable quantitative avant et
    après retraitement des données manquantes sous forme d'histogramme

    Arguments:
    --------------------------------
    df_before: dataframe avant imputation des données manquantes, obligatoire
    var : str : variable à analyser
    df_after : dataframe après imputation des données manquantes, obligatoire
    bins : int : nombre d'intervalles, 100 par défaut

    return:
    --------------------------------
    None
    """
    plt.figure(figsize=(20, 5))
    ax1 = plt.subplot(121)
    plt.title(f'Distribution de la variable {var} \n avant imputation des valeurs manquantes', fontsize=15)
    ax1.hist(df_before[var], bins=bins, color='#b8b8d2')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    draw_text_hist(df_before, var, ax1)

    ax2 = plt.subplot(122)
    plt.title(f'Distribution de la variable {var} \n après imputation des valeurs manquantes', fontsize=15)
    ax2.hist(df_after[var], bins=bins, color='#b8b8d2')
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    draw_text_hist(df_after, var, ax2)

    plt.show()
    plt.close()


def distrib_imput_nan_quali(df_before, var, df_after):
    """Fonction qui affiche la distribution d'une variable qualitative avant et
    après retraitement des données manquantes sous forme de diagramme en barre

    Arguments:
    --------------------------------
    df_before: dataframe avant imputation des données manquantes, obligatoire
    var : str : variable à analyser, obligatoire
    df_after : dataframe après imputation des données manquantes, obligatoire

    return:
    --------------------------------
    None
    """
    plt.figure(figsize=(20, 5))
    plt.subplot(121)
    plt.title(f'Distribution de la variable {var} \n avant imputation des valeurs manquantes', fontsize=15)
    modalites = df_before[var].value_counts()
    modalites[0:15].plot.bar(color='#b8b8d2', edgecolor='black')
    plt.xticks(rotation=45, ha='right', fontsize=15)
    plt.yticks(fontsize=15)

    plt.subplot(122)
    plt.title(f'Distribution de la variable {var} \n après imputation des valeurs manquantes', fontsize=15)
    modalites = df_after[var].value_counts()
    modalites[0:15].plot.bar(color='#b8b8d2', edgecolor='black')
    plt.xticks(rotation=45, ha='right', fontsize=15)
    plt.yticks(fontsize=15)

    plt.show()
    plt.close()


def plot_tx_remplissage(df, fig):
    """Fonction qui affiche le taux de remplissage de chaque variable
    sous forme de barplot

    Arguments:
    --------------------------------
    df: dataframe: tableau en entrée, obligatoire
    fig : taille de la figure

    return:
    --------------------------------
    None"""

    not_nan_percent = round(df.notna().mean().sort_values(ascending=False), 4) * 100
    ax = sns.barplot(y=not_nan_percent.index, x=not_nan_percent.values, palette='Purples_r')
    plt.title("Taux de remplissage des variables")
    plt.ylabel("")
    plt.xlabel("Taux de remplissage")
    etiquette_h(ax)
    plt.show()


# --------------------------------------------------------------------
# --------------------------- CORRELATIONS ---------------------------
# --------------------------------------------------------------------


def pairs_corr(df, liste_var_quanti, seuil=0.5):
    """Fonction qui affiche le taux de corrélation (Pearson)
    par paire de variables supérieur au seuil défini en entrée

    Arguments:
    --------------------------------
    df: dataframe: tableau en entrée, obligatoire
    liste_var_quanti : list : liste des variables numériques continues
    seuil : float : seuil minimum de corrélation, 50% par défaut

    return:
    --------------------------------
    set des variables uniques pour lesquelles la distribution avant
    et après doit être vérifiée"""

    # Matrice de corrélation pour les variables continues
    corr = df[liste_var_quanti].corr()
    # Dataframe avec les pairs de variables et leur corrélation
    df_corr = corr.stack().reset_index().sort_values(0, ascending=False)
    # Zip des pairs
    df_corr['pairs'] = list(zip(df_corr.level_0, df_corr.level_1))
    # set index pairs
    df_corr.set_index(['pairs'], inplace=True)
    # Suppression des levels
    df_corr.drop(columns=['level_1', 'level_0'], inplace=True)
    # Renommage colonne en "correlation"
    df_corr.columns = ['correlation']
    # Suppression des doublons avec conservation des corrélations de pairs >= x
    df_corr.drop_duplicates(inplace=True)
    df_corr = df_corr[(abs(df_corr.correlation) >= seuil) & (abs(df_corr.correlation) != 1)]
    # Affichage
    display(df_corr)
    liste_var = list(df_corr.index)
    a, b = list(zip(*liste_var))
    set_var_a_tester = set(list(a + b))

    return set_var_a_tester


# --------------------------------------------------------------------
# ------------------- FEATURE ENGINEERING ----------------------------
# --------------------------------------------------------------------

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


# --------------------------------------------------------------------
# ---------------------- FONCTIONS PLOTS ---------------------------- 
# --------------------------------------------------------------------

def etiquette_v(ax, espace=5):
    """Ajoute les étiquettes en haut de chaque barre sur un barplot vertical.

    Arguments:
    --------------------------------
    ax: (matplotlib.axes.Axes): objet matplotlib contenant les axes
    du plot à annoter.
    espace : int : distance entre les étiquettes et les barres

    return:
    --------------------------------
    None
    """

    # Pour chaque barre, placer une étiquette
    for rect in ax.patches:
        # Obtenir le placement de X et Y de l'étiquette à partir du rectangle
        y_value = rect.get_height()
        x_value = rect.get_x() + rect.get_width() / 2

        # Espace entre la barre et le label
        space = espace
        # Alignement vertical
        va = 'bottom'

        # Si la valeur est négative, placer l'étiquette sous la barre
        if y_value < 0:
            # Valeur opposer de l'argument espace
            space *= -1
            # Alignement vertical 'top'
            va = 'top'

        # Utiliser la valeur Y comme étiquette et formater avec 0 décimale
        label = "{:.0f}".format(y_value)

        # Créer l'annotation
        ax.annotate(
            label,
            (x_value, y_value),
            xytext=(0, space),
            textcoords="offset points",
            ha='center',
            va=va)


def etiquette_h(ax):
    """Ajoute les étiquettes en haut de chaque barre sur un barplot horizontal.

    Arguments:
    --------------------------------
    ax: (matplotlib.axes.Axes): objet matplotlib contenant les axes
    du plot à annoter.

    return:
    --------------------------------
    None
    """

    for p in ax.patches:
        etiquette = '{:,.0f}'.format(p.get_width())
        width, height = p.get_width(), p.get_height()
        x = p.get_x() + width + 0.02
        y = p.get_y() + height / 2
        ax.annotate(etiquette, (x, y))

    # --------------------------------------------------------------------


# ----------------- PLOTS VARIABLES QUANTITATIVES --------------------
# -------------------------------------------------------------------- 

def draw_text_hist(df, var, ax):
    """Fonction qui permet d'afficher du texte dans chaque histogramme.

    Arguments:
    --------------------------------
    df: dataframe: tableau en entrée, obligatoire
    var: str: variable
    ax: (matplotlib.axes.Axes): objet matplotlib contenant les axes
    du plot à annoter.

    return:
    --------------------------------
    None
    """
    at = AnchoredText(
        f'Moy = {round(df[var].mean(), 2)} \nMed = {round(df[var].median(), 2)} \nStd = {round(df[var].std(), 2)}',
        loc=1, prop=dict(size=12), frameon=True)
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax.add_artist(at)


def univariate_plots_hist(df, liste_col_quanti, nb_lignes, nb_col, nb_bins, fig):
    """Représentation par histogramme des variables quantitatives

    Arguments:
    --------------------------------
    df: dataframe: tableau en entrée, obligatoire
    liste_col_quanti : liste : liste des variables quantitatives, obligatoire
    nb_lignes : int : nombre de lignes
    nb_col : int : nombre de plots par colonne
    nb_bins : int : nombre de bins
    fig : taille de la figure

    return:
    --------------------------------
    None
    """

    for i, c in enumerate(liste_col_quanti, 1):
        ax = fig.add_subplot(nb_lignes, nb_col, i)
        ax.hist(df[c], bins=nb_bins, color='#b8b8d2')
        ax.set_title(c, fontsize=10)
        ax.title.set_fontweight('bold')
        plt.xticks(rotation=45, ha='right')

    plt.tight_layout(w_pad=2, h_pad=2)
    plt.show()
    plt.close()


def univariate_plots_box(df, liste_col_quanti, nb_lignes, nb_col, fig, outliers=False):
    """Représentation par boxplot des variables quantitatives

    Arguments:
    --------------------------------
    df: dataframe: tableau en entrée, obligatoire
    liste_col_quanti : liste : liste des variables quantitatives, obligatoire
    nb_lignes : int : nombre de lignes
    nb_col : int : nombre de plots par colonne
    showfliers : bool : affichage des outliers ou non (pas d'affichage par défaut)
    fig : taille de la figure

    return:
    --------------------------------
    None
    """
    for i, c in enumerate(liste_col_quanti, 1):
        ax = fig.add_subplot(nb_lignes, nb_col, i)
        ax = sns.boxplot(data=df, x=c, showfliers=outliers, color='#b8b8d2')
        ax.set_title(c)
        ax.title.set_fontweight('bold')
        plt.xticks(rotation=45, ha='right')

    plt.tight_layout(w_pad=2, h_pad=2)
    plt.show()
    plt.close()


def bivariate_kdeplots(df, var_quali, liste_var_quanti, nb_lignes, nb_col, palette):
    """KDE plot permettant de visualiser les distributions multiples des variables quantitatives
    en fonction d'une variable qualitative en légende.


    Arguments:
    --------------------------------
    df: dataframe: tableau en entrée, obligatoire
    var_quali : str : variable sur laquelle effectuer le "hue"
    liste_var_quanti : liste : liste des variables quantitatives, obligatoire
    nb_lignes : int : nombre de lignes
    nb_col : int : nombre de plots par colonne

    return:
    --------------------------------
    None
    """
    fig = plt.figure(figsize=(15, 20))
    for i, c in enumerate(liste_var_quanti, 1):
        ax = fig.add_subplot(nb_lignes, nb_col, i)
        ax = sns.kdeplot(data=df, x=c, hue=var_quali, multiple="stack", palette=palette)
        ax.set_title(c)
        ax.title.set_fontweight('bold')
        plt.xticks(rotation=45, ha='right')

    plt.tight_layout(w_pad=2, h_pad=2)
    plt.show()
    plt.close()


# --------------------------------------------------------------------
# ----------------- PLOTS VARIABLES CATEGORIELLES --------------------
# --------------------------------------------------------------------


def categ_distrib_plot(df, liste_categ_col, nb_lignes, nb_col, fig):
    """Barplot de distribution des modalités des variables catégorielles avec le nombre de
    modalités en titre.

    Arguments:
    --------------------------------
    df: dataframe: tableau en entrée, obligatoire
    liste_categ_col : liste : liste des variables catégorielles, obligatoire
    nb_lignes : int : nombre de lignes
    nb_col : int : nombre de plots par colonne
    fig : taille de la figure

    return:
    --------------------------------
    None
    """
    for i, c in enumerate(liste_categ_col, 1):
        ax = fig.add_subplot(nb_lignes, nb_col, i)
        modalites = df[c].value_counts()
        n_modalites = modalites.shape[0]

        if n_modalites > 15:
            modalites[0:15].plot.bar(color='#b8b8d2', edgecolor='black', ax=ax)

        else:
            modalites.plot.bar(color='#b8b8d2', edgecolor='black')

        ax.set_title(f'{c} \n ({n_modalites} modalités)', fontweight='bold')
        label = [item.get_text() for item in ax.get_xticklabels()]
        short_label = [lab[0:10] + '.' if len(lab) > 10 else lab for lab in label]
        ax.axes.set_xticklabels(short_label)
        plt.xticks(rotation=45, ha='right')

    plt.tight_layout(w_pad=2, h_pad=2)


def bivariate_plots_box(df, var_quali, liste_var_quanti, nb_lignes, nb_col, fig, outliers=False):
    """Barplot de distribution des modalités des variables catégorielles avec le nombre de
    modalités en titre.

    Arguments:
    --------------------------------
    df: dataframe: tableau en entrée, obligatoire
    liste_categ_col : liste : liste des variables catégorielles, obligatoire
    nb_lignes : int : nombre de lignes
    nb_col : int : nombre de plots par colonne
    fig : taille de la figure

    return:
    --------------------------------
    None
    """
    for i, c in enumerate(liste_var_quanti, 1):
        ax = fig.add_subplot(nb_lignes, nb_col, i)
        meanprops = {'marker': 'o', 'markeredgecolor': 'black', 'markerfacecolor': 'firebrick'}
        ax = sns.boxplot(data=df, y=c, x=var_quali, showfliers=outliers, showmeans=True,
                         meanprops=meanprops, palette=['powderblue', 'tomato'])
        plt.suptitle(f'Dispersion des variables en fonction des {var_quali}s', fontsize=16,
                     fontweight='bold')

        ax.title.set_fontweight('bold')
        labels = [item.get_text() for item in ax.get_xticklabels()]
        short_labels = [l[0:20] + '.' if len(l) > 0 else l for l in labels]
        ax.axes.set_xticklabels(short_labels)
        plt.xticks([0, 1], ['Non Défaillant', 'Défaillant'])
        plt.xlabel('')

    plt.tight_layout(w_pad=2, h_pad=2)
    plt.show()
    plt.close()


def bivariate_barplots_categ(df, var_x, var_hue):
    """Barplot de distribution des modalités des variables catégorielles
    en fonction d'une autre variable catégorielle.

    Arguments:
    --------------------------------
    df: dataframe: tableau en entrée, obligatoire
    var_x : str : variable catégorielle en abscisse, obligatoire
    var_hue : str : autre variable catégorielle (en légende), obligatoire

    return:
    --------------------------------
    None
    """
    fig = plt.figure(figsize=(15, 5))

    # Table de contingence
    print(f'{var_x} vs {var_hue}')
    tab_cont = pd.crosstab(df[var_x], df[var_hue])
    if len(np.where(tab_cont <= 5)[0]) == 0:
        print('Chaque effectif de la table de contingence >= 5 => Test du Chi2 applicable')
        # Running Chi2 test
        st_chi2, st_p, st_dof, st_exp = chi2_contingency(tab_cont)
        if st_p < 0.05:
            print(f"pvalue: {st_p} < 0.05 => on rejette H0, les variables sont dépendantes")
        else:
            print(f"pvalue: {st_p} > 0.05 => on accepte H0, les variables sont indépendantes")
        print('-------------------------------------------------------------------')
    else:
        print("Au moins un effectif de la table de contingence < 5 => Test du Chi2 d'indépendance")
        # Running Chi2 Indépendance
        expected, observed, stat = pg.chi2_independence(df, var_x, var_hue)
        if stat['pval'][0] < 0.05:
            print(f"pvalue: {stat['pval'][0]} < 0.05 => on rejette H0, les variables sont dépendantes")
        else:
            print(f"pvalue: {stat['pval'][0]} > 0.05 => on accepte H0, les variables sont indépendantes")
        print('-------------------------------------------------------------------')

    # Premier graph
    ax = fig.add_subplot(121)
    sns.countplot(data=df, x=var_x, hue=var_hue, palette=['powderblue', 'tomato'])
    legend_handles, _ = ax.get_legend_handles_labels()
    ax.legend(legend_handles, ['Non défaillant', 'Défaillant'],
              bbox_to_anchor=(1, 1),
              title='')
    plt.xticks(rotation=45, ha='right')
    plt.title(f'{var_x} \n Nombre par TARGET', fontweight='bold', fontsize=12)

    # Deuxième graph
    ax1 = fig.add_subplot(122)
    # Calcul du pourcentage
    tab_cont_prop = pd.crosstab(df[var_x], df[var_hue], normalize="index")
    tab_cont_prop['total'] = tab_cont_prop.sum(axis=1)
    tab_cont_prop = tab_cont_prop.reset_index()

    # bar chart 1 → top bars (non défaillants)
    bar1 = sns.barplot(x=var_x, y='total', data=tab_cont_prop, color='powderblue')

    # bar chart 2 → bottom bars (défaillants)
    bar2 = sns.barplot(x=var_x, y=1, data=tab_cont_prop, color='tomato')

    # add legend
    top_bar = mpatches.Patch(color='powderblue', label='Non Défaillant')
    bottom_bar = mpatches.Patch(color='tomato', label='Défaillant')
    plt.legend(handles=[top_bar, bottom_bar])
    plt.xticks(rotation=45, ha='right')

    plt.xlabel("", fontsize=12)
    plt.ylabel("%", fontsize=12)

    plt.title(f'{var_x} \n Part des défaillants et non défaillants', fontweight='bold', fontsize=12)

    plt.show()
    plt.close()
