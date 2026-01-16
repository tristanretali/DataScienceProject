import pandas as pd
import config as cfg


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fait le nettoayge nécessaire déduit de l'EDA

    Args:
        df (pd.DataFrame): Notre dataset

    Returns:
        pd.DataFrame: Le dataset nettoyé
    """

    df.dropna(inplace=True)
    df = df.drop_duplicates()

    df = change_data_types(df)

    # Correction du type de Fuel pour les Tesla
    df.loc[df["Brand"] == "Tesla", "Fuel Type"] = "Electric"
    # Conversion du prix en euros
    df["Price"] = (df["Price"] * 0.86).round(2)

    # Suppression des colonnes qui ne sont pas utile pour le modèle
    df = df.drop(columns=["Car ID", "Brand"])

    df = remove_price_outliers(df)

    df = define_labels(df)

    return df


def change_data_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Changement du type de certaines colonnes

    Args:
        df (pd.DataFrame): Notre dataset

    Returns:
        pd.DataFrame: Le dataset avec les types changés
    """
    # Conversion en category
    for current_column in df.select_dtypes(include=["object"]).columns:
        df[current_column] = df[current_column].astype("category")

    # Conversion en int de certaines colonnes
    df["Year"] = df["Year"].astype(int)
    df["Mileage"] = df["Mileage"].astype(int)

    return df


def remove_price_outliers(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Suppression des valeurs aberrantes de la colonne 'Price'

    Args:
        df (pd.DataFrame): Notre dataset

    Returns:
        pd.DataFrame: Le dataset avec les valeurs aberrantes en moins
    """
    # Calcule des quartiles
    Q1 = df["Price"].quantile(0.25)
    Q3 = df["Price"].quantile(0.75)
    IQR = Q3 - Q1

    # Définition des intervalles
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Création d'un masque, chaque valeur aberrante est marquée True
    is_outlier = (df["Price"] < lower_bound) | (df["Price"] > upper_bound)

    # Calcule de la médiane pour chaque modèle de voiture
    median_by_model = df.groupby("Model")["Price"].median()

    # Remplacement des valeurs aberrantes par la médiane du modèle
    for idx in df[is_outlier].index:
        model = df.loc[idx, "Model"]
        df.loc[idx, "Price"] = median_by_model[model]

    return df


def define_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Changement des prix dans la tranche qui leur correspond

    Args:
        df (pd.DataFrame): Notre dataset

    Returns:
        pd.DataFrame: Le dataset avec le changement du prix des voitures en intervalle
    """
    max_price = df["Price"].max()
    # Création de notre liste de bons avec un intervalle de 5000. On va jusqu'à la valeur maximale.
    bins = list(range(0, int(max_price) + 5000, 5000))
    # Création de nos labels qui sont composées de la limite inférieur et supérieur de chaque bin
    labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins) - 1)]

    # On associe chaque prix à la catégorie correspondante
    df["Price"] = pd.cut(df["Price"], bins=bins, labels=labels, right=True)

    return df


if __name__ == "__main__":
    df = pd.read_csv(f"{cfg.DATA_DIR}/car_price.csv")
    df_cleaned = clean_data(df)

    print(df_cleaned.head())

    # TODO Faire entraînement et sauvegarde des modèles
