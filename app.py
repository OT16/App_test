"""
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
"""
# PIL (Python Imaging Library) is a free library for the Python programming language that adds support for opening, manipulating, and saving many different image file formats.
from PIL import Image
import streamlit as st
import numpy as np
#import plotly.express as px
import pandas as pd
#import plotly.io as pio
#from neuralprophet import NeuralProphet

# Load the image >>>>>>>>>>>>
image_nyc = Image.open('newyorkcity.jpg')
# Display the NYU logo on the Streamlit app
st.image(image_nyc, width=100)

# Set the title for the Streamlit app >>>>>>>>>>>>
st.title("New York Data Explorer")

# Create a sidebar header and a separator
st.sidebar.header("Dashboard")
st.sidebar.markdown("---")
app_mode = st.sidebar.selectbox('🔎 Select Page',['Introduction','Visualization','Prediction'])
 
# read the dataset using the compression zip
data = pd.read_csv('https://cdn-charts.streeteasy.com/rentals/All/medianAskingRent_All.zip?_ga=2.25329597.105908559.1708960519-800494817.1708960519',compression='zip')
df = data.melt(id_vars=['areaName','Borough','areaType'], var_name='Attribute', value_name='Value')

#The below renames df to data, and is done because we're removing #NA results
data = df.dropna()

if app_mode == "Introduction":
  st.sidebar.markdown("## Welcome!")
  st.sidebar.markdown("You don't have anything selected yet.\n Make a selection on the map to explore how demographic, housing, and quality of life characteristics\n compare across neighborhoods and demographic groups over the past two decades.\n Make your selection by community district*, borough, or city.\n Or, switch to the Displacement Risk Map and select a neighborhood to see the level of risk residents face \n of being unable to remain in their homes or neighborhoods.\n")
  st.markdown("## Introduction")

elif app_mode == "Visualization":
  symbols = df.columns
  st.markdown(df.columns)
  st.bar_chart(data=df, x=symbols[1],y=symbols[-1], width=0, height=0, use_container_width=True)

elif app_mode == "Prediction":
  st.markdown("## Prediction")

