import numpy as np
import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
np.random.seed(1234)
st.title("DOCUMENTO")
datos=np.random.normal(0,1, size=(100,4))
data=pd.DataFrame(datos,columns=list("ABCD"))
st.dataframe(data)
E=np.random.normal(0,1, size=100)
y=data["A"]*2+data["B"]*3+data["C"]*4+data["D"]*0.3+10 
model=DecisionTreeRegressor(max_depth=4)
model.fit(data,y)
st.subheader("A")
val_A=st.slider("selecciones el valor de A",
          data["A"].min(),
          data["A"].max())
st.subheader("B")
val_B=st.slider("selecciones el valor de B",
          data["B"].min(),
          data["B"].max())
st.subheader("C")
val_C=st.slider("selecciones el valor de C",
          data["C"].min(),
          data["C"].max())
st.subheader("D")
val_D=st.slider("selecciones el valor de D",
          data["D"].min(),
          data["D"].max())

valores=np.array([[val_A, val_B, val_C, val_D]])
pre=model.predict(valores)
st.write(pre)