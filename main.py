import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from sklearn import preprocessing, linear_model, metrics, model_selection
import warnings


def main():
    warnings.filterwarnings("ignore")

    df = pd.read_csv("heart.csv")

    # Turning all object values to a string
    string_cols = df.select_dtypes(include="object").columns
    df[string_cols] = df[string_cols].astype("string")

    # # Plot showing how each data point correlates to the existence of heart disease
    # fig = px.imshow(df.corr(), title="Correlation Plot of the Heart Disease Prediction")
    # fig.show()
    #
    # # Heart disease distribution between male and female
    # fig = px.histogram(df,
    #                    x="HeartDisease",
    #                    color="Sex",
    #                    hover_data=df.columns,
    #                    title="Heart Disease Distribution by Sex",
    #                    barmode="group")
    # fig.show()
    #
    # # Ratio of male to female participants in the dataset
    # fig = px.histogram(df,
    #                    x="Sex",
    #                    hover_data=df.columns,
    #                    title="Sex Ratio Data")
    # fig.show()
    #
    # # Distribution of chest pain types within the dataset
    # fig = px.histogram(df,
    #                    x="ChestPainType",
    #                    color="Sex",
    #                    hover_data=df.columns,
    #                    title="Chest Pain Type Distribution")
    # fig.show()
    #
    # # Distribution of max heart rate within the dataset
    # fig = px.histogram(df,
    #                    x="MaxHR",
    #                    color="Sex",
    #                    hover_data=df.columns,
    #                    title="Max Heart Rate Levels Distribution",)
    # fig.update_layout(bargap=0.2)
    # fig.show()
    #
    # # Distribution of cholesterol levels within the dataset
    # fig = px.histogram(df,
    #                    x="Cholesterol",
    #                    color="Sex",
    #                    hover_data=df.columns,
    #                    title="Cholesterol Levels Distribution",)
    # fig.update_layout(bargap=0.2)
    # fig.show()
    #
    # # Distribution of resting blood pressure levels within the dataset
    # fig = px.histogram(df,
    #                    x="RestingBP",
    #                    color="Sex",
    #                    hover_data=df.columns,
    #                    title="Resting Blood Pressure Levels Distribution",)
    # fig.show()
    #
    # #
    # plt.figure(figsize=(15, 10))
    # hue = "HeartDisease"
    # g = sns.pairplot(df, hue=hue, palette="husl")
    # plt.title("Looking for Insights in Data")
    # plt.legend(title=hue, loc="lower right", handles=g._legend_data.values())
    # plt.tight_layout()
    # plt.show()
    #
    # #
    # plt.figure(figsize=(15, 10))
    # for i, col in enumerate(df.columns, 1):
    #     plt.subplot(4, 3, i)
    #     plt.title(f"Distribution of {col} Data")
    #     sns.histplot(df[col], kde=True)
    #     plt.tight_layout()
    # plt.show()

    # Converting categorical data into dummy values
    df_preped = pd.get_dummies(df, columns=string_cols, drop_first=False)

    # Setting the target heart disease column to the end
    df_preped.drop("HeartDisease", axis=1, inplace=True)
    df_preped = pd.concat([df_preped, df["HeartDisease"]], axis=1)

    # Using min max scaling to normalize data
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(df_preped)
    df_minmax = scaler.transform(df_preped)
    df_minmax = pd.DataFrame(df_minmax, columns=df_preped.columns)

    x = df_minmax.iloc[:, :-1]
    y = df_minmax.iloc[:, -1]
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=0.3)

    # Logistic Regression Model
    logistic_regression_model = linear_model.LogisticRegression(max_iter=1000)
    logistic_regression_model.fit(x_train, y_train)
    logistic_regression_prediction = logistic_regression_model.predict(x_test)
    print(metrics.accuracy_score(y_test, logistic_regression_prediction))
    print(metrics.precision_score(y_test, logistic_regression_prediction))
    print(metrics.recall_score(y_test, logistic_regression_prediction))
    print(metrics.f1_score(y_test, logistic_regression_prediction))


if __name__ == '__main__':
    main()
