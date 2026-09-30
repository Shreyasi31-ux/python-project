#Project status- ONGOING
import numpy as np
import pandas as pd

df1 = pd.read_csv(r"C:\Users\user\Downloads\MachineLearningCSV\MachineLearningCVE\Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv")
df2 = pd.read_csv(r"C:\Users\user\Downloads\MachineLearningCSV\MachineLearningCVE\Friday-WorkingHours-Morning.pcap_ISCX.csv")
df3 = pd.read_csv(r"C:\Users\user\Downloads\MachineLearningCSV\MachineLearningCVE\Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv")

dataset = pd.concat([df1, df2, df3], ignore_index=True)

dataset.columns = dataset.columns.str.strip()

dataset = dataset.sample(
    n=min(5000, len(dataset)),
    random_state=0
).reset_index(drop=True)

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

X = X.apply(pd.to_numeric, errors="coerce")

X.replace(
    [np.inf, -np.inf],
    np.nan,
    inplace=True
)

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="mean")

X = imputer.fit_transform(X)

X = np.nan_to_num(
    X,
    nan=0.0,
    posinf=0.0,
    neginf=0.0
)

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

encoder = LabelEncoder()

y = encoder.fit_transform(y)

scaler = StandardScaler()

X = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split
from sklearn.decomposition import KernelPCA

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score

results = []

test_sizes = [0.2, 0.4, 0.6]

kernels = [
    "rbf",
    "cosine",
    "sigmoid"
]

for test_size in test_sizes:

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=0,
        stratify=y
    )

    for kernel in kernels:

        kpca = KernelPCA(
            n_components=2,
            kernel=kernel,
            gamma=15,
            eigen_solver="arpack"
        )

        X_train_new = kpca.fit_transform(X_train)

        X_test_new = kpca.transform(X_test)

        models = [
            (
                "Algorithm 1 - Linear Regression",
                LinearRegression()
            ),
            (
                "Algorithm 3 - Ridge Regression",
                Ridge(alpha=1.0)
            ),
            (
                "Algorithm 5 - Elastic Net",
                ElasticNet(
                    alpha=0.01,
                    l1_ratio=0.5,
                    max_iter=5000
                )
            ),
            (
                "Algorithm 13 - KNN",
                KNeighborsClassifier(
                    n_neighbors=5
                )
            ),
            (
                "Algorithm 15 - SVM",
                SVC(
                    kernel="rbf",
                    random_state=0
                )
            )
        ]

        for name, model in models:

            model.fit(
                X_train_new,
                y_train
            )

            prediction = model.predict(
                X_test_new
            )

            if name.startswith("Algorithm 1"):
                prediction = np.rint(prediction)
                prediction = np.clip(
                    prediction,
                    0,
                    len(encoder.classes_) - 1
                )

            elif name.startswith("Algorithm 3"):
                prediction = np.rint(prediction)
                prediction = np.clip(
                    prediction,
                    0,
                    len(encoder.classes_) - 1
                )

            elif name.startswith("Algorithm 5"):
                prediction = np.rint(prediction)
                prediction = np.clip(
                    prediction,
                    0,
                    len(encoder.classes_) - 1
                )

            accuracy = accuracy_score(
                y_test,
                prediction
            )

            results.append(
                {
                    "Algorithm": name,
                    "Test Size": test_size,
                    "Kernel": kernel,
                    "Accuracy": accuracy
                }
            )

            print(
                "{} | Test Size = {} | Kernel = {} | Accuracy = {:.4f}".format(
                    name,
                    test_size,
                    kernel,
                    accuracy
                )
            )

result_df = pd.DataFrame(results)

print("\nTOTAL DIFFERENT RESULTS =", len(result_df))

print("\n45 DIFFERENT RESULTS\n")

print(
    result_df.to_string(
        index=False
    )
)

result_df.to_csv(
    "45_Different_Results.csv",
    index=False
)

print(
    "\n45_Different_Results.csv created successfully."
)
