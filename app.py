import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import pickle
import Recommend
sn.set_style("darkgrid")
# Loading the Machine Learning Model
model = pickle.load(open("Crop_Predictor.sav","rb"))

st.title("Crop Recommendation Web App")
st.subheader("Improve your farming strategy by knowing which crop will be better for your farm !!")
st.image("farming.jpg")
data = pd.read_csv("Crop_recommendation.csv")
data.rename({"N":"Nitrogen","P":"Phosphorus","K":"Potassium","humidity":"%humidity","label":"Crops"},axis = 1,inplace=True)
dc = {}
columns = [i for i in data]
columns.pop()
crop = list(data["Crops"].unique())
var = 1
for i in crop:
    dc[i] = var
    var += 1

st.subheader("The Dataset: ")
st.write(data)
st.sidebar.header("Analytics Section :")
st.sidebar.subheader("Find Correlation between different features for better decisions !!")
x1 = st.sidebar.selectbox("Select the first feature: ",columns)
x2 = st.sidebar.selectbox("Select the second feature: ",columns)

# Find Correlation
correlation = round(data[x1].corr(data[x2]),4)
st.header(f"The correlation between {x1} and {x2} is: {correlation} ")
# Choose a plot
st.sidebar.subheader("Visualize your data: ")
select = st.sidebar.selectbox("Plots",( "Scatterplot","KDE plot","Histogram","Correlation Matrix"))
color = st.sidebar.selectbox("Select color ",( "violet","purple","limegreen","orange"))
category = st.sidebar.selectbox("Select crop wise or not ",("Crops",None))
st.sidebar.subheader("How each features are affecting growth of crops: ")
factor = st.sidebar.selectbox("Select a feature: ",columns)
st.title("Data Visualization: ")
st.set_option('deprecation.showPyplotGlobalUse', False)

if select == "Scatterplot":
    sn.scatterplot(x = x1,y = x2,data = data,color = color,hue = category)
    plt.legend(loc = 'lower right')
    plt.show()
    st.pyplot()

if select == "KDE plot":
    sn.kdeplot(x = data[x1],color = color)
    st.subheader(f"The skewness of {x1} is {round(data[x1].skew(),4)}")
    plt.show()
    st.pyplot()

if select == "Histogram":
    sn.displot(x=data[x1], color=color)
    plt.show()
    st.pyplot()

if select == "Correlation Matrix":
    sn.heatmap(data.corr(),annot = True,cmap = "viridis")
    plt.show()
    st.pyplot()



st.header(f"How growth of crops is changing wrt {factor}")
plt.barh(data["Crops"],data[factor],color = color)
plt.show()
st.pyplot()
st.set_option('deprecation.showPyplotGlobalUse', False)

# Input the features:
st.header("Give input features to know which crop is best for your farm: ")
N = st.number_input(label="Nitrogen")
P = st.number_input(label="Phosphorous")
K = st.number_input(label="Potassium")
temp = st.number_input(label="Temperature (0C)")
hum = st.number_input(label="% Humidity")
ph = st.number_input(label="pH value")
rain = st.number_input(label="Rain Fall (mm)")

array = [[N,P,K,temp,hum,ph,rain]]
val = model.predict(array)
st.header("Recommendation Of Best Crops:")
st.write("-----------------------------------")
for i in dc:
    if dc[i] == val:
        st.header(f"The recommended crops for the above conditions: ")
        lst = list(set(Recommend.Recommend(i)))
        
        for i in lst:
            st.subheader(i.upper())
