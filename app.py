
# Streamlit is an open-source app framework for Machine Learning and Data Science projects.
import streamlit as st

# Pandas is a fast, powerful, flexible, and easy-to-use open-source data analysis and data manipulation Python library.
import pandas as pd

# NumPy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions.
import numpy as np

# Matplotlib is a plotting library for the Python programming language and its numerical mathematics extension NumPy.
#import matplotlib.pyplot as plt

# Seaborn is a Python data visualization library based on Matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics.
#import seaborn as sns

# Plotly Express is a terse, consistent, high-level API for creating figures with Plotly.py.
#import plotly.express as px

# (Note: Streamlit is imported twice in the provided code, which is redundant.)
import streamlit as st

# Python's built-in library for generating random numbers.
import random

# PIL (Python Imaging Library) is a free library for the Python programming language that adds support for opening, manipulating, and saving many different image file formats.
from PIL import Image


# Load the NYU logo image >>>>>>>>>>>>
image_nyu = Image.open('nyu.png')
# Display the NYU logo on the Streamlit app
st.image(image_nyu, width=100)

# Set the title for the Streamlit app >>>>>>>>>>>>
st.title("Chocolate Review Analysis")

# Create a sidebar header and a separator
st.sidebar.header("Dashboard")
st.sidebar.markdown("---")
app_mode = st.sidebar.selectbox('🔎 Select Page',['Introduction','Visualization','Prediction'])

df = pd.read_csv("chocolate.csv")


#list_variables = [" ".join(x.split("_")).title() for x in df.columns]
list_variables = df.columns

# Display a header for the Visualization section
st.markdown("## Visualization")
symbols = st.multiselect("Select two variables", list_variables, ["rating", "cocoa_percent"])

df["cocoa_percent"] = df["cocoa_percent"].apply(lambda x : x[:-1])
df["cocoa_percent"] = df["cocoa_percent"].apply(lambda x : float(x))
rating_min, rating_max = st.sidebar.slider('Select Rating Range', min_value=int(df['rating'].min()), max_value=int(df['rating'].max()), value=(int(df['rating'].min()), int(df['rating'].max())))
cocoa_percent_min, cocoa_percent_max = st.sidebar.slider('Select Cocoa Percent Range', min_value=float(df['cocoa_percent'].min()), max_value=float(df['cocoa_percent'].max()), value=(float(df['cocoa_percent'].min()), float(df['cocoa_percent'].max())))

# Filtering the dataframe based on the slider values
filtered_df = df[(df['rating'] >= rating_min) & (df['rating'] <= rating_max) & (df['cocoa_percent'] >= cocoa_percent_min) & (df['cocoa_percent'] <= cocoa_percent_max)]


tab1, tab2 = st.tabs(["Line Chart", "Bar Chart"])

tab1.subheader("Line Chart")
# Display a line chart for the selected variables
tab1.line_chart(data=filtered_df, x=symbols[0], y=symbols[1], width=0, height=0, use_container_width=True)

tab2.subheader("Bar Chart")
# Display a bar chart for the selected variables
tab2.bar_chart(data=filtered_df, x=symbols[0], y=symbols[1], use_container_width=True)

