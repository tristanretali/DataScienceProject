import pandas as pd
import config as cfg
import pickle
from sklearn.preprocessing import (
    StandardScaler,
    OrdinalEncoder,
    OneHotEncoder,
    LabelEncoder,
)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


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


def preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Étape de preprocessing de notre dataset

    Args:
        df (pd.DataFrame): Notre dataset nettoyé

    Returns:
        tuple: Renvoie les variables dont on a besoin pour lancer notre pipeline
    """
    # Définition de nos labels et de la variable cible
    X = df.drop("Price", axis=1)
    y = df["Price"]

    # Encodage de la variable cible
    price_label_encoder = LabelEncoder()
    y = price_label_encoder.fit_transform(y)

    # Features numérique (int, float)
    numeric_features = ["Year", "Engine Size", "Mileage"]
    # Features catégorielles
    categorical_features_non_ord = ["Fuel Type", "Transmission"]
    categorical_features_ord = ["Condition", "Model"]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_non_ord_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    categorical_ord_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ordinay", OrdinalEncoder()),
        ]
    )

    # Initialsation de l'étape de preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            (
                "cat_non_ord",
                categorical_non_ord_transformer,
                categorical_features_non_ord,
            ),
            ("cat_ord", categorical_ord_transformer, categorical_features_ord),
        ],
        remainder="drop",
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Sauvegarde le l'encodage des labels pour les retrouver lorsqu'on va faire la prédiction
    with open(f"{cfg.DATA_DIR}/label_encoder.pkl", "wb") as f:
        pickle.dump(price_label_encoder, f)

    return preprocessor, X_train, y_train


def launch_pipeline(
    preprocessor,
    X_train: pd.DataFrame,
    y_train,
    mode: str = "v1",
):
    """


    Args:
        preprocessor (_type_): Le preprocessing à utiliser sur nos différentes features
        X_train (pd.DataFrame): Dataset d'entraînement
        y_train (_type_): Labels d'entraînement
        mode (str, optional): Version du modèle à entraîner. Defaults to "v1".
    """
    if mode == "v1":
        pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    RandomForestClassifier(n_estimators=1000, random_state=42),
                ),
            ]
        )
    elif mode == "v2":
        pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    GradientBoostingClassifier(n_estimators=100, random_state=42),
                ),
            ]
        )

    pipeline.fit(X_train, y_train)

    with open(f"{cfg.MODELS_DIR}/model_{mode}.pkl", "wb") as f:
        pickle.dump(pipeline, f)


def train_models():
    """
    Entraînement de nos modèles
    """
    df = pd.read_csv(f"{cfg.DATA_DIR}/car_price.csv")
    df_cleaned = clean_data(df)
    preprocessor, X_train, y_train = preprocessing(df_cleaned)

    # Entraînement de deux modèles avec des algorithmes différents
    launch_pipeline(preprocessor, X_train, y_train, mode="v1")
    launch_pipeline(preprocessor, X_train, y_train, mode="v2")


if __name__ == "__main__":
    train_models()
