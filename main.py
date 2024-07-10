import streamlit as st
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

st.title('Polynomial Regression Visualizer')
st.write(":blue[***See Your Model's Fit Evolve!***]")
st.write("Explore the impact of polynomial degree on regression models with this interactive visualization tool. "
         "Adjust the polynomial degree to see real-time changes in model performance and fit, helping you to "
         "understand the optimal complexity for your data.")


@st.cache_resource
def generate_training_test_data():
    # Generate Independent and Dependent Features and get Train and Test data
    X = 6 * np.random.rand(100, 1) - 3
    y = 0.5 + X ** 2 + 1.5 * X + 2 + np.random.randn(100, 1)

    X_new_data = np.linspace(-3, 3, 200).reshape(200, 1)
    y_new_data = 0.5 + X_new_data ** 2 + 1.5 * X_new_data + 2 + np.random.randn(200, 1)
    X_train_data, X_test_data, y_train_data, y_test_data = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train_data, X_test_data, y_train_data, y_test_data, X_new_data, y_new_data


def poly_regression(degree):

    poly_features = PolynomialFeatures(degree=degree, include_bias=True)
    lin_reg = LinearRegression()
    poly_reg = Pipeline([
        ("poly_features", poly_features),
        ("lin_reg", lin_reg)
    ])
    poly_reg.fit(X_train, y_train)
    y_predict_new = poly_reg.predict(X_new)
    score = round(r2_score(y_new, y_predict_new), 4)
    st.subheader('Model Performance:')
    st.metric(label="R2 Score", value=score)

    # Plotting Prediction Line
    plt.plot(X_new, y_predict_new, 'r', label='Degree' + str(degree), linewidth=3)
    plt.scatter(X_train, y_train, color="b")
    plt.scatter(X_test, y_test, color="g")
    plt.legend(loc="upper left")
    plt.xlabel("X")
    plt.ylabel("Y")

    # Use Streamlit to display the plot
    st.subheader('Plot')
    st.pyplot(plt)


# Generate Training and Test data
X_train, X_test, y_train, y_test, X_new, y_new = generate_training_test_data()
st.subheader('Select Polynomial Degree:')
degree_option = st.number_input('Select Polynomial Degree', min_value=0,
                                max_value=20, value=1, step=1, label_visibility='collapsed')
poly_regression(degree_option)
