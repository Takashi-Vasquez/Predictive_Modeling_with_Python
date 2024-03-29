import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle

def modelo_entrenamiento(datos,Tsize,Rstate,Baraja,Nngs,C,G,EPS):

    # datos = pd.read_excel("Post_datos_planificacionHassPeru(20101-20212).xlsx")
    #test_size
    if Tsize is not None:
        VTS= Tsize
    else:
        VTS= 0.3
    #random_state
    if Rstate is not None:
        VRstate= Rstate
    else:
        VRstate= 10
    #shuffle
    if Baraja is not None:
        VBaraja= Baraja
    else:
        VBaraja= True
    #n_neighbors
    if Nngs is not None:
        VNngs= int(Nngs)
    else:
        VNngs= 4
    #C
    if C is not None:
        VC= C
    else:
        VC= 1.0
    #gamma
    if G is not None:
        VG= str(G)
    else:
        VG= 'auto'
    #epsilon
    if EPS is not None:
        VEPS= EPS
    else:
        VEPS= 2

    X=datos[['anio','campania','fundo_nro','superficie_sembrada_ha','edad_planta','Temp_Semestral']].values
    y=datos['superficie_cosechada_ha'].values

    X_e,X_p,Y_e,Y_p=train_test_split(X,y,test_size=VTS, random_state=VRstate,shuffle=VBaraja)

    #modelos
    knn=KNeighborsRegressor(n_neighbors=VNngs)
    rl = LinearRegression()
    svm=SVR(kernel ='linear',C=VC,gamma=VG ,epsilon=VEPS)


    #Ajustamos el modelo
    m_knn=knn.fit(X_e,Y_e)
    m_svm=svm.fit(X_e,Y_e)
    m_rl=rl.fit(X_e,Y_e)

    #porcentaje de prediccion
    scoreKNN=round(knn.score(X_e, Y_e) * 100, 2)
    scoreSVM=round(svm.score(X_e, Y_e) * 100, 2)
    scoreRL=round(rl.score(X_e, Y_e) * 100, 2)
    print(round(rl.score(X_e, Y_e) * 100, 2))
    print(round(knn.score(X_e, Y_e) * 100, 2))
    print(round(svm.score(X_e, Y_e) * 100, 2))


    ArrScore = {"KNN": round(knn.score(X_e, Y_e) * 100, 2), "SVM": round(svm.score(X_e, Y_e) * 100, 2), "RL": round(rl.score(X_e,Y_e)*100, 2)}

    #Guardando modelos de forma binaria
    datasets=pickle.dump(datos, open('Archivos Binarios/dataSets.pkl', 'wb'))
    score=pickle.dump(ArrScore, open('Archivos Binarios/score.pkl', 'wb'))
    modelo_knn = pickle.dump(m_knn, open('Archivos Binarios/modeloKNN.pkl', 'wb'))
    modelo_svm =pickle.dump(m_svm, open('Archivos Binarios/modeloSVM.pkl', 'wb'))
    modelo_rl=pickle.dump(m_rl, open('Archivos Binarios/modeloRL.pkl', 'wb'))

    return scoreKNN,scoreRL,scoreSVM,datasets,score,modelo_knn,modelo_svm,modelo_rl
