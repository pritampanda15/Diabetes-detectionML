#import the libraries
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier, MultiOutputRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, precision_score, recall_score, matthews_corrcoef, classification_report, recall_score, precision_recall_curve
from sklearn.metrics import accuracy_score
from PIL import Image
import streamlit as st

st.write ("""
#Diabetes Detection
Detect if someone has diabetes using Random forest classifier
""")
#open and display
image = Image.open('image.jpeg')
st.image(image, caption='ML', use_column_width=True)

#read the data
df = pd.read_csv('diabetes.csv')

#set subheader
st.subheader('Data Information:')

#show the table
st.dataframe(df)
#show statistics
st.write(df.describe())
chart =st.bar_chart(df)

#split the data into independent variables
X = df.iloc[:, 0:8].values
Y = df.iloc[:,-1].values
#split the data into 75% training and 25% test
X_train, X_test, Y_train, Y_test = train_test_split (X, Y, test_size =0.25, random_state=0)

#Get the feature input
def get_user_input():
    pregnancies = st.sidebar.slider('pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('glucose', 0, 199, 117)
    blood_pressure = st.sidebar.slider('blood_pressure', 0, 122, 72)
    skin_thickness = st.sidebar.slider('skin_thickness', 0, 99, 23)
    insulin = st.sidebar.slider('insulin', 0.0, 846.0, 30.5)
    BMI = st.sidebar.slider('BMI', 0.0, 67.1, 32.0)
    DPF = st.sidebar.slider('DPF', 0.078, 2.42, 0.3725)
    age = st.sidebar.slider('age', 21, 81, 29)

    #store a dictonary into a variable
    user_data = {'pregnancies': pregnancies,
                'glucose': glucose,
                'blood_pressure': blood_pressure,
                'skin_thickness': skin_thickness,
                'insulin': insulin,
                'BMI': BMI,
                'DPF': DPF,
                'age': age
                 }
 #Transform data into data frame
    features = pd.DataFrame(user_data, index =[0])
    return  features
#Store the userinput into a variable
user_input = get_user_input()

#set a subheader and display users input
st.subheader('User Input:')
st.write(user_input)

#Create and train the model
RandomForestClassifier=RandomForestClassifier()
RandomForestClassifier.fit(X_train, Y_train)

#Show the model metrics
st.subheader('Model Test Accuracy Score:')
st.write(str(accuracy_score(Y_test, RandomForestClassifier.predict(X_test)) * 100)+'%' )

#store the model predictions in a variable
prediction = RandomForestClassifier.predict(user_input)

#set a subheader and display the classification
st.subheader('Classification: ')
st.write(prediction)