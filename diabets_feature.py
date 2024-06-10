# FEATURE ENGINEERING & MACHINE LEARNING ILE DIYABET TAHMINI

# Bu proje, Amerika Birleşik Devletleri'nde bulunan Pima Indian kadınları üzerinde yapılan diyabet araştırması için
# kullanılan veri seti üzerinde çeşitli veri analizleri, feature engineering ve machine learning modellemesi içerir.
# Projenin amacı, hamilelik sayısı, glikoz seviyeleri, vücut kitle indeksi gibi çeşitli özellikler kullanılarak bir
# kişinin diyabet hastalığına sahip olup olmadığını belirlemektir.

# VERI SETI
# Veri seti ABD'deki Ulusal Diyabet-Sindirim-Böbrek Hastalıkları Enstitüleri'nde tutulan büyük veri setinin parçasıdır.
# ABD'deki Arizona Eyaleti'nin en büyük 5. şehri olan Phoenix şehrinde yaşayan 21 yaş ve üzerinde olan Pima Indian
# kadınları üzerinde yapılan diyabet araştırması için kullanılan verilerdir. Hedef değişken "outcome" olarak belirtilmiş
# olup; 1 diyabet test sonucunun pozitif oluşunu, 0 ise negatif oluşunu belirtmektedir.

# DEGISKENLER
# Pregnancies: Hamilelik sayısı
# Glucose: Oral glikoz tolerans testinde 2 saatlik plazma glikoz konsantrasyonu
# Blood Pressure:
# Kan Basıncı: (Küçük tansiyon) (mm Hg)
# SkinThickness: Cilt Kalınlığı
# Insulin: 2 saatlik serum insülini (mu U/ml)
# DiabetesPedigreeFunction: Fonksiyon (Oral glikoz tolerans testinde 2 saatlik plazma glikoz konsantrasyonu)
# BMI: Vücut kitle endeksi
# Age: Yaş (yıl)
# Outcome: Hastalığa sahip (1) ya da değil (0)

# VERİ SETİNE GENEL BAKIŞ

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import seaborn as sns
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)
pd.set_option('display.width', 500)

df = pd.read_csv("/Users/ahmetbozkurt/Desktop/Diabets_Feature_Engineering/dataset/diabetes.csv")
df.head()
df.shape

# VERİ SETİNE GENEL BAKIŞ
def check_df(dataframe, head=5):
    print("-" * 25 + "Shape" + "-" * 25)
    print(dataframe.shape)
    print("-" * 25 + "Types" + "-" * 25)
    print(dataframe.dtypes)
    print("-" * 25 + "The First data" + "-" * 25)
    print(dataframe.head(head))
    print("-" * 25 + "The Last data" + "-" * 25)
    print(dataframe.tail(head))
    print("-" * 25 + "Missing values" + "-" * 25)
    print(dataframe.isnull().sum())
    print("-" * 25 + "Describe the data" + "-" * 25)
    # Sayısal değişkenlerin dağılım bilgisi
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)
    print("-" * 25 + "Distinct Values" + "-" * 25)
    print(dataframe.nunique())
check_df(df)

# Değişkenleri kendi aralarında sınıflandıralım.
def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    Veri setindeki kategorik,numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
    Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir

    Parameters
    ----------
    dataframe: dataframe
        Değişken isimleri alınmak istenen dataframe'dir
    cat_th: int, float
        Numerik fakat kategorik değişkenler için sınıf eşik değeri
    car_th: int, float
        Kategorik fakat kardinal değişkenlerin sınıf eşik değeri

      Returns
    ------
        cat_cols: list
                Kategorik değişken listesi
        num_cols: list
                Numerik değişken listesi
        cat_but_car: list
                Kategorik görünümlü kardinal değişken listesi

    Examples
    ------
        import seaborn as sns
        df = sns.load_dataset("iris")
        print(grab_col_names(df))


    Notes
    ------
        cat_cols + num_cols + cat_but_car = toplam değişken sayısı
        num_but_cat cat_cols'un içerisinde.
        Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı
    """

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].dtypes != "O" and dataframe[col].nunique() < cat_th]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].dtypes == "O" and dataframe[col].nunique() > car_th]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car
cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Her değişkenin analizlerine bakalım.

# Kategorik Değişkenler
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
    print("##########################################")
    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show(block=True)

for col in cat_cols:
    cat_summary(df, col, plot=True)

# Numerik Değişkenler
def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        sns.histplot(dataframe[numerical_col], bins=30, kde=True)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)

for col in num_cols:
    num_summary(df, col, plot=True)


# Numerik Değişkenlere Göre Hedef Değişken Analizi.
def target_numerical(dataframe, target, col_numerical):
    print(dataframe.groupby(target).agg({col_numerical: "mean"}))
    print("#" * 40)

for col in num_cols:
    target_numerical(df, "Outcome", col)

# Aykırı Gözlemlere Bakalım.
def outlier_thresholds(dataframe, col_name, q1=0.01, q3=0.99):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low, up = outlier_thresholds(dataframe, col_name)
    if dataframe.loc[(dataframe[col_name] < low) | (dataframe[col_name] > up)].any(axis=None):
        return True
    else:
        return False

def replace_with_thresholds(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    dataframe.loc[dataframe[col_name] > up_limit, col_name] = up_limit
    dataframe.loc[dataframe[col_name] < low_limit, col_name] = low_limit

for col in num_cols:
    print(col, check_outlier(df, col))

# Eksik Gözlem Analizi Yapalım.
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, ratio], axis=1, keys=["n_miss", "ratio"])
    print(missing_df)
    if na_name:
        return na_columns

missing_values_table(df)  # Eksik Değer Yok.

# Korelasyon Analizi Yapalım.
def corr_map(df, width=14, height=6, annot_kws=15, corr_th=0.7):
    corr = df.corr()
    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(
        np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))  # np.bool yerine bool
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]
    mtx = np.triu(df.corr())
    f, ax = plt.subplots(figsize=(width, height))
    sns.heatmap(df.corr(),
                annot=True,
                fmt=".2f",
                ax=ax,
                vmin=-1,
                vmax=1,
                cmap="RdBu",
                mask=mtx,
                linewidth=0.4,
                linecolor="black",
                annot_kws={"size": annot_kws})
    plt.yticks(rotation=0, size=15)
    plt.xticks(rotation=75, size=15)
    plt.title('\nCorrelation Map\n', size=40)
    plt.show()
    return drop_list

corr_map(df)


# FEATURE ENGINEERING
df.columns = [col.title() for col in df.columns]
df.head()

cat_cols, num_cols, cat_but_car = grab_col_names(df)

# Veri setimizde eksik değer yok fakat bazı değişkenlerde içerisinde minimum değeri 0 olan satırlar var.Bunlar bir
# bir insanın hayatta olması için asla 0 olmaması gereken değerler.O yüzden bunlara eksik değer olarak bakıp ilgili
# bunları doldurmalıyız.

df.describe().T
nan_cols = [col for col in df.columns if col not in ["Pregnancies", "Outcome"]]
nan_cols = [col for col in nan_cols if df.loc[df[col] == 0].shape[0] > 0]

for col in nan_cols:
    print(col, df.loc[df[col] == 0].shape[0])

for col in nan_cols:
    df[col] = df[col].apply(lambda x: np.nan if x == 0 else x)

# İkinci bir yöntem
for col in nan_cols:
    df[col] = np.where(df[col] == 0, np.nan, df[col])

missing_values_table(df)
df[nan_cols].describe().T

# Eksik Değerlerin Bağımlı Değişken ile İlişkisinin İncelenmesi
def missing_vs_target(dataframe, target, na_columns):
    temp_df = dataframe.copy()
    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)
    na_flags = temp_df.loc[: ,temp_df.columns.str.contains("_NA_")].columns
    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")

missing_vs_target(df, "Outcome", nan_cols)

# Boş değer barındıran değişkenlerin betimsel istatistiklerine bakalım.
for col in nan_cols:
    print("#" * 10 + col + "#" * 10)
    print(df.groupby("Outcome")[col].describe().T)

# Eksik değerleri atamadan önce değişikliklere bakmak için yeni bir değişken oluşturalım.
df["Insulin_Flag"] = df["Insulin"]  # Daha sonra silindi.
df.head()

# Eksik Değerleri Median İle Doldurmak Veri Seti Açısından bizim için daha yararlı olacaktır.
for col in nan_cols:
    df[col] = df[col].fillna(df.groupby("Outcome")[col].transform("median"))

missing_values_table(df)

# Aykırı değerleri tekrar kontrol edelim.
for col in num_cols:
    print(col, check_outlier(df, col))  # Aykırı Değer Yok


# Yeni Feature'lar Üretelim.
# Bmi Değişkeninden aralıklar oluşturalım.
df["New_Bmi_Index"] = pd.cut(df["Bmi"], bins=[0, 18.5, 24.9, 29.9, 34.9, 39.9, float("inf")],
                             labels=["UNDERWEIGHT", "NORMAL", "OVERWEIGHT", "OBESE_1", "OBESE_2", "EXTREMLY OBESE"])

# Age değişkeni kategorileştirme
df["New_Age_Category"] = pd.cut(df["Age"], bins=[0, 18, 30, 45, 65, 75, 85, float("inf")],
                                labels=["Çocuk", "Genç", "Genç-Yetişkin", "Yetişkin", "Genç-Yaşlı", "Yaşlı", "Çok-Yaşlı"])

# Insulin Değişkeninden Kategori Oluşturma
df["New_Insulin_Category"] = pd.cut(df["Insulin"], bins=[0, 16, 166, float("inf")],
                                    labels=["LOW_INSULIN", "NORMAL", "ANORMAL"])

# Glucose Değişkeninden Kategori oluşturma
df["New_Glucose_Category"] = pd.cut(df["Glucose"], bins=(0, 70, 99, 126, float("inf")),
                                    labels=["LOW_GLUCOSE", "NORMAL_GLUCOSE", "SECRET_GLUCOSE", "HIGH_GLUCOSE"])

# Bloodpressure Değişkeninden Kategori Oluşturma
df["New_Blood_Category"] = pd.cut(df["Bloodpressure"], bins=[0, 60, 80, 90, 100, float("inf")],
                                  labels=["HIPOTANSIYON", "NORMAL", "PREHIPERTANSIYON", "EVRE_1", "EVRE_2"])


# Pregnancies Değişkeninden Kategori Oluşturma
df["New_Pregnancies_Cat"] = pd.cut(df["Pregnancies"], bins=[0, 1, 3, 6, float("inf")], right=False,
                                   labels=["Hamile_Yok", "Normal_Hamile", "Çok_Hamile", "Aşırı_Hamile"])

# Pregnancies Değişkeninden Özellik Üretme
df["New_Preg_Absurd"] = df.apply(lambda x: 1 if x["Age"] <= 40 and x["Pregnancies"] > 3 else 0, axis=1)

# Encoding İşlemlerini Gerçekleştirelim.
cat_cols, num_cols, cat_but_car = grab_col_names(df)
cat_cols = [col for col in cat_cols if col not in "Outcome"]

# Label Encoder için ikili değişken yok.
binary_col = [col for col in cat_cols if df[col].nunique() == 2 and df[col].dtypes in ["object", "category"]]

# One Hot Encoder yapalım.
ohe_cols = [col for col in cat_cols if df[col].nunique() > 2 and df[col].dtypes in ["object", "category"]]

def one_hot_encoder(dataframe, categorical_col, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_col, drop_first=drop_first, dtype=int)
    return dataframe

df = one_hot_encoder(df, ohe_cols)
df.head()
df.info()

# Standartlaştırma İşlemini Yapalım.
rs = RobustScaler()
df[num_cols] = rs.fit_transform(df[num_cols])

# Model Aşamasına Geçelim.
models = [("LR", LogisticRegression()),
          ("CART", DecisionTreeClassifier()),
          ("KNN", KNeighborsClassifier()),
          ("GBM", GradientBoostingClassifier()),
          ("RF", RandomForestClassifier()),
          ("XGBoost", XGBClassifier()),
          ("LightGBM", LGBMClassifier(verbosity=-1)),]

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

for name, model in models:
    cv_results = cross_validate(model, X, y, cv=10, scoring=["accuracy", "f1", "recall", "precision", "roc_auc"])
    print(f"########## {name} ##########")
    print(f"Accuracy: {round(cv_results['test_accuracy'].mean(), 4)}")
    print(f"Auc: {round(cv_results['test_roc_auc'].mean(), 4)}")
    print(f"Recall: {round(cv_results['test_recall'].mean(), 4)}")
    print(f"Precision: {round(cv_results['test_precision'].mean(), 4)}")
    print(f"F1: {round(cv_results['test_f1'].mean(), 4)}")

########## LR ##########
# Accuracy: 0.8384
# Auc: 0.91
# Recall: 0.757
# Precision: 0.7754
# F1: 0.7642
########## CART ##########
# Accuracy: 0.8581
# Auc: 0.847
# Recall: 0.81
# Precision: 0.7945
# F1: 0.7988
########## KNN ##########
# Accuracy: 0.8385
# Auc: 0.8868
# Recall: 0.7426
# Precision: 0.7867
# F1: 0.7616
########## GBM ##########
# Accuracy: 0.879
# Auc: 0.956
# Recall: 0.799
# Precision: 0.8511
# F1: 0.8201
########## RF ##########
# Accuracy: 0.8802
# Auc: 0.9497
# Recall: 0.7944
# Precision: 0.8528
# F1: 0.8172
########## XGBoost ##########
# Accuracy: 0.8841
# Auc: 0.9538
# Recall: 0.8137
# Precision: 0.8501
# F1: 0.8281
########## LightGBM ##########
# Accuracy: 0.8843
# Auc: 0.9522
# Recall: 0.8215
# Precision: 0.8459
# F1: 0.8311
########## CatBoost ##########
# Accuracy: 0.8842
# Auc: 0.9542
# Recall: 0.814
# Precision: 0.8539
# F1: 0.8268

# LightGBM ile model değerlendirme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
lgbm = LGBMClassifier(verbosity=-1)
lgbm.fit(X_train, y_train)
y_pred = lgbm.predict(X_test)
print(classification_report(y_test, y_pred))

# Heatmap olarak confusion matrix'i görselleştirelim.
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d")
plt.xlabel('Tahmin Edilen Sınıf')
plt.ylabel('Gerçek Sınıf')
plt.title('Confusion Matrix')
plt.show()

# Modele en çok etki eden feature'ları görelim.
def plot_importance(model, features, num=10, save=False):
    if num is None:
        num = len(features.columns)
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 6))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value", ascending=False).iloc[:num])
    plt.title('Feature Importances')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

plot_importance(lgbm, X)
