import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle

def modelo_entrenamiento():

    datos = pd.read_excel("Post_datos_planificacionHassPeru(20101-20212).xlsx")

    X=datos[['anio','campania','fundo_nro','superficie_sembrada_ha','edad_planta','Temp_Semestral']].values
    y=datos['superficie_cosechada_ha'].values

    X_e,X_p,Y_e,Y_p=train_test_split(X,y,test_size=0.3, random_state=0)

    #modelos
    knn=KNeighborsRegressor(n_neighbors=4)
    svm=SVR()
    rl=LinearRegression()

    #Ajustamos el modelo
    m_knn=knn.fit(X_e,Y_e)
    m_svm=svm.fit(X_e,Y_e)
    m_rl=rl.fit(X_e,Y_e)

    #porcentaje de prediccion
    print(round(knn.score(X_e, Y_e) * 100, 2))
    print(round(svm.score(X_e, Y_e) * 100, 2))
    print(round(rl.score(X_e,Y_e)*100, 2))

    ArrScore = {"KNN": round(knn.score(X_e, Y_e) * 100, 2), "SVM": round(svm.score(X_e, Y_e) * 100, 2), "RL": round(rl.score(X_e,Y_e)*100, 2)}

    #Guardando modelos de forma binaria
    pickle.dump(datos,open('dataSets.pkl','wb'))
    pickle.dump(ArrScore,open('score.pkl','wb'))
    pickle.dump(m_knn,open('modeloKNN.pkl','wb'))
    pickle.dump(m_svm,open('modeloSVM.pkl','wb'))
    pickle.dump(m_rl,open('modeloRL.pkl','wb'))



modelo_entrenamiento()