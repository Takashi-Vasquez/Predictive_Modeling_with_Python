import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import pickle
import datetime
import time
from numpy import round

modelo_knn = pickle.load(open('modeloKNN.pkl', 'rb'))
modelo_svm = pickle.load(open('modeloSVM.pkl', 'rb'))
modelo_rl  = pickle.load(open('modeloRL.pkl', 'rb'))
score      = pickle.load(open('score.pkl','rb'))
datasets   = pickle.load(open('dataSets.pkl','rb'))

def prediccion(input_arr,menuopciones):
    if menuopciones == 'Regresion lineal':
        st.write("seleccionaste regresion")
        prediccion_result = modelo_rl.predict(input_arr)
    elif menuopciones == 'KNN':
        st.write("seleccionaste KNN")
        prediccion_result =modelo_knn.predict(input_arr)
    elif menuopciones == 'SVM':
        st.write("seleccionaste SVM")
        prediccion_result =modelo_svm.predict(input_arr)
    return prediccion_result

def st_interface():
    st.set_page_config( page_title = "HassPerú" , page_icon = "🥑" )
    st.title("SISTEMA PREDICTIVO PARA LA PLANIFICACIÓN DE CULTIVO")
    imagen=Image.open("Image/HassPeru.png")
    st.image(imagen, caption='Frescura, calidad y sabor', width=700)
    st.markdown("<p style='text-align: left; color: #789d10;'>Hass Perú es una empresa agroindustrial con más de "
                "10 años de experiencia en el sector, dedicada al cultivo y comercialización de"
                "paltas Hass y arándano a nivel nacional e internacional.</p>", unsafe_allow_html=True)
    st.markdown("Web Site: [HASS PERÚ](http://www.hassperu.com/es/)")


    # Selectores Panel izquierdo
    opciones = ['Regresion lineal', 'KNN', 'SVM']
    menuopciones = st.sidebar.selectbox('Seleccione el modelo a utilizar', opciones)
    st.sidebar.text("Precisión:")
    if menuopciones == 'Regresion lineal':
        if score["RL"] >75:
            st.sidebar.success(str(score["RL"])+" %")
        elif score["RL"] >25 and score["RL"] < 75:
            st.sidebar.warning(str(score["RL"])+" %")
        elif score["RL"] < 25:
            st.sidebar.error(str(score["KNN"])+" %")
    elif menuopciones == 'KNN':
        if score["KNN"] > 75:
            st.sidebar.success(str(score["KNN"])+" %")
        elif score["KNN"] > 25 and score["KNN"] < 75:
            st.sidebar.warning(str(score["KNN"])+" %")
        elif score["KNN"] < 25:
            st.sidebar.error(str(score["KNN"])+" %")
    elif menuopciones == 'SVM':
        if score["SVM"] > 75:
            st.sidebar.success(str(score["SVM"])+" %")
        elif score["SVM"] > 25 and score["SVM"] < 75:
            st.sidebar.warning(str(score["SVM"])+" %")
        elif score["SVM"] < 25:
            st.sidebar.error(str(score["SVM"])+" %")
    # Importacion de datos
    uploaded_file = st.sidebar.file_uploader("Cargue el Archivo", type=['xlsx', 'csv'],
                                                 help="Puede seleccionar archivo xlsx/csv para su predicción")
    col1, col2, col3 = st.columns(3)
    if uploaded_file is None:
        time.sleep(2)
        #Selectores Panel DERECHO
        st.subheader("CONDICIONES ")

        ##extrayendo año actual
        date = datetime.date.today()
        year = date.strftime("%Y")
        st.text("Año : " + year)

        ##Seleccion de campaña
        campaña= st.radio("Seleccionar Campaña: ",(1, 2))

        ##Opciones N° de Fundo
        fundoN=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
        fundopciones = st.selectbox('Seleccione el N° de fundo', fundoN)

        ##Superficie Sembrada
        superf_Sembrada = st.number_input('Superficie Sembrada por Hectárea', min_value=0, max_value=50, value=20)

        ##Edada de las plantas
        edad_Planta = st.number_input('Edad de la Planta', min_value=0, max_value=14, value=5)
        ##Temperatura Promedio
        Temp = st.slider('Temperatura Promedio?', 10.0, 30.0, 20.0)

        # Envia las entradas al modelo.
        if st.button("Predecir"):
            input_arr = np.array([year, campaña, fundopciones, superf_Sembrada,edad_Planta,Temp]).reshape(1, -1)
            resultadoPredicción = prediccion(input_arr, menuopciones)
            st.write('Superficie Cosechada:     ',round(resultadoPredicción[0],2))


    #prediccion de los datos cargados
    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file, dtype={'superficie_cosechada_ha': np.float64,
                                                'produccion_tm': np.float64,
                                                'rendimiento_kgxha': np.float64,
                                                'cant_abonoOrg(kg)': np.float64,
                                                'abonoTotal(kg)': np.float64},
                                                na_values = np.nan)
        #IMPUTACIÓN DE LOS DATOS
        # st.text(df.isnull().sum())
        df= df.fillna(0)
        # time.sleep(2)
        st.markdown(
            '<h3 style="text-align:center; font-family:arial;color:white">Datos Importados</h3>',unsafe_allow_html=True)
        st.dataframe(df.style.format({'superficie_cosechada_ha': '{:.2f}',
                         'produccion_tm': '{:.2f}',
                         'rendimiento_kgxha': '{:.2f}',
                         'cant_abonoOrg(kg)': '{:.2f}',
                         'abonoTotal(kg)': '{:.2f}'}).set_properties(**{'text-align': 'center'}),1000)


        if st.button("Predecir"):
            independiente=df[['anio','campania','fundo_nro','superficie_sembrada_ha','edad_planta','Temp_Semestral']]

            df['superficie_cosechada_ha'] = round(prediccion(independiente, menuopciones),1)
            with st.spinner('Espere por favor...'):
                # time.sleep(5)
                #COLUMNA SUPERFICIE COSECHADA
                N_Fundo = df['fundo_nro']
                conditionlist = [
                    (N_Fundo == 1),(N_Fundo == 2),
                    (N_Fundo == 3),(N_Fundo == 4),
                    (N_Fundo == 5),(N_Fundo == 6),
                    (N_Fundo == 7),(N_Fundo == 8),
                    (N_Fundo == 9),(N_Fundo == 10),
                    (N_Fundo == 11),(N_Fundo == 12),
                    (N_Fundo == 13),(N_Fundo == 14),
                    (N_Fundo == 15),(N_Fundo == 16),
                    (N_Fundo == 17)]
                superf_C=df['superficie_cosechada_ha']
                choicelist = [superf_C*18.09 , superf_C*17.51
                              , superf_C*17.45, superf_C*18.17
                              , superf_C*17.77, superf_C*19.06
                              , superf_C*18.00, superf_C*17.30
                              , superf_C*16.72, superf_C*17.41
                              , superf_C*17.86, superf_C*18.28
                              , superf_C*17.09, superf_C*16.64
                              , superf_C*16.65, superf_C*18.00
                              ,superf_C*18.33]
                df['produccion_tm'] = np.select(conditionlist, choicelist, default='Not Specified')
                df['produccion_tm']=round(df['produccion_tm'].astype(float),1)
                df['rendimiento_kgxha']=df['produccion_tm'] * 1000
                df['rendimiento_kgxha'] = df['rendimiento_kgxha'].astype(int)

                #COLUMNA CANTIDAD DE ABONO EN kg
                edad_planta=df['edad_planta']
                conditionEdad = [
                    (edad_planta > 12) ,
                    (edad_planta == 12) | (edad_planta == 11),
                    (edad_planta == 10),
                    (edad_planta == 9),
                    (edad_planta == 8) | (edad_planta == 7),
                    (edad_planta == 6) | (edad_planta == 5),
                    (edad_planta == 4) | (edad_planta == 3),
                    (edad_planta == 2),
                    (edad_planta == 1)]
                choicelistEdad = [80,60,55,50,45,40,35,25,20]
                df['cant_abonoOrg(kg)'] = np.select(conditionEdad, choicelistEdad, default='Not Specified')
                df['cant_abonoOrg(kg)'] = df['cant_abonoOrg(kg)'].astype(int)
                df['abonoTotal(kg)'] = df['cant_abonoOrg(kg)']* df['cant_planta']
                df['Temp_Semestral'] = round(df['Temp_Semestral'],1)
                #REPORTE
                st.markdown(
                    '<h3 style="background-color:MediumSeaGreen; text-align:center; font-family:arial;color:white">REPORTE</h3>',
                    unsafe_allow_html=True)
                st.dataframe(df.style.format({'produccion_tm': '{:.2f}',
                                    'superficie_cosechada_ha': '{:.2f}',
                                    'Temp_Semestral': '{:.1f}'}).\
                             set_properties(**{'text-align': 'center'}).\
                             highlight_max(subset=['superficie_cosechada_ha','produccion_tm'],color='green').\
                             highlight_min(subset=['superficie_cosechada_ha', 'produccion_tm'], color='red')
                            ,1000)

            combineData = pd.concat([df,datasets], axis=0)
            @st.cache
            def convert_df(df):
                # IMPORTANT: Cache the conversion to prevent computation on every rerun
                return df.to_csv(index=False).encode('utf-8')

            dato = convert_df(combineData)
            col1,col2,col3 = st.columns(3)
            col2.download_button(
            label = "Descargar Archivo en Excel",
            data = dato,
            file_name = 'datos_planificacion.csv',
            mime='text/csv',
            )
if __name__ == "__main__":
    st_interface()
