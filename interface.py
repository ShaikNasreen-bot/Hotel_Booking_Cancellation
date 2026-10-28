import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import  RandomForestClassifier
from sklearn.metrics import classification_report
from xgboost import XGBClassifier


st.title("🏨Hotel Booking cancellation",text_alignment='center')

data=pd.read_csv(r"ML_Report.csv")
data['is_cancelled']=data['is_cancelled'].map({"No":0,"Yes":1})
st.markdown("""
<style>
p, label, span, div {
    letter-spacing: 0.5px;
    word-spacing: 4px;
}
</style>
""", unsafe_allow_html=True)
# print(data.columns)
X=data[['Hotel_Type','Lead_Time_Days','Room_Type','Previous_Bookings', 'Previous_Cancellations', 'Booking_Channel', 'Deposit_Type']]

X['Hotel_Type']=X['Hotel_Type'].map({'City Hotel':1, 'Resort Hotel':2, 'Unknown':0})
X['Room_Type']=X['Room_Type'].map({'Unknown':0, 'Deluxe':3, 'Suite':2, 'Standard':1})
X['Booking_Channel']=X['Booking_Channel'].map({'Corporate':3, 'Online':1, 'Offline':0, 'Travel Agent':2})
X['Deposit_Type']=X['Deposit_Type'].map({'Non-Refundable':1, 'No Deposit':0 ,'Refundable':2})

y=data['is_cancelled']

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=42)

model=XGBClassifier(max_depth=3,n_estimator=550,learning_rate=0.50)

model.fit(x_train,y_train)

HT = st.number_input(
    "Enter hotel type 0 - Unknown, 1 - City Hotel, 2 - Resort Hotel",
    min_value=0, value=0, step=1, format="%d"
)

ND = st.number_input(
    "Enter no. of days stayed",
    min_value=0, value=0, step=1, format="%d"
)

RT = st.number_input(
    "Enter room type 0-Unknown, 1-SD, 2-Suite, 3-Deluxe",
    min_value=0, value=0, step=1, format="%d"
)

PB = st.number_input(
    "Enter no. booking",
    min_value=0, value=0, step=1, format="%d"
)

PC = st.number_input(
    "Enter no. cancellation",
    min_value=0, value=0, step=1, format="%d"
)

BC = st.number_input(
    "Enter your channel 0-Offline, 1-Online, 2-TA, 3.Corporate",
    min_value=0, value=0, step=1, format="%d"
)

DT = st.number_input(
    "Enter Deposit type 0-No Deposit, 1-NOR, 2-Refundable",
    min_value=0, value=0, step=1, format="%d"
)

bt = st.button("Submit")

if bt:
    output = model.predict([[HT, ND, RT, PB, PC, BC, DT]])

    if output[0] == 1:
        st.write("Cancelled")
    else:
        st.write("Not Cancelled")

train_output=model.predict(x_train)
test_output=model.predict(x_test)

train_report=classification_report(y_train,train_output)
test_report=classification_report(y_test,test_output)
st.write(train_report)
st.write(test_report)