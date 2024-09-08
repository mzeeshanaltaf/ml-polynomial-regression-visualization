from util import *

st.title('Polynomial Regression Visualizer')
st.write(":blue[***See Your Model's Fit Evolve!***]")
st.write("Explore the impact of polynomial degree on regression models with this interactive visualization tool. "
         "Adjust the polynomial degree to see real-time changes in model performance and fit, helping you to "
         "understand the optimal complexity for your data.")

st.subheader('Select Polynomial Degree:')
degree_option = st.number_input('Select Polynomial Degree', min_value=0,
                                max_value=20, value=1, step=1, label_visibility='collapsed')

# Draw the polynomial regression plot
poly_regression(degree_option)

# Display footer
display_footer()
