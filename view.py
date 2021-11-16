from datetime import  datetime
import time
import streamlit as st
from PIL import Image
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
from numpy import round
import app
from app import *
import SENAMHI


modelo_knn = pickle.load(open('modeloKNN.pkl', 'rb'))
modelo_svm = pickle.load(open('modeloSVM.pkl', 'rb'))
modelo_rl  = pickle.load(open('modeloRL.pkl', 'rb'))
score      = pickle.load(open('score.pkl','rb'))
datasets   = pickle.load(open('dataSets.pkl','rb'))
Titulo, detalle,tabla,link = SENAMHI.webScraping()
now = datetime.now()

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

def sidebar():
    # Selectores Panel izquierdo
    opciones = ['Regresion lineal', 'KNN', 'SVM']
    global menuopciones
    menuopciones = st.sidebar.selectbox('Seleccione el modelo a utilizar', opciones)
    st.sidebar.text("Precisión:")
    if menuopciones == 'Regresion lineal':
        if score["RL"] > 75:
            st.sidebar.success(str(score["RL"]) + " %")
        elif score["RL"] > 25 and score["RL"] < 75:
            st.sidebar.warning(str(score["RL"]) + " %")
        elif score["RL"] < 25:
            st.sidebar.error(str(score["KNN"]) + " %")
    elif menuopciones == 'KNN':
        if score["KNN"] > 75:
            st.sidebar.success(str(score["KNN"]) + " %")
        elif score["KNN"] > 25 and score["KNN"] < 75:
            st.sidebar.warning(str(score["KNN"]) + " %")
        elif score["KNN"] < 25:
            st.sidebar.error(str(score["KNN"]) + " %")
    elif menuopciones == 'SVM':
        if score["SVM"] > 75:
            st.sidebar.success(str(score["SVM"]) + " %")
        elif score["SVM"] > 25 and score["SVM"] < 75:
            st.sidebar.warning(str(score["SVM"]) + " %")
        elif score["SVM"] < 25:
            st.sidebar.error(str(score["SVM"]) + " %")
    return menuopciones


#HOME PAGE
def home():
# if pagina == "Home":
    st.title("Los mejores frutos de nuestra tierra by HassPerú")
    imagen=Image.open("Image/HassPeru.png")
    st.image(imagen, caption='Frescura, calidad y sabor', width=700)
    st.markdown("<p style='text-align: left; color: #789d10;'>Hass Perú es una empresa agroindustrial con más de "
                "10 años de experiencia en el sector, dedicada al cultivo y comercialización de"
                "paltas Hass y arándano a nivel nacional e internacional.</p>", unsafe_allow_html=True)
    st.markdown("Web Site: [HASS PERÚ](http://www.hassperu.com/es/)")
#Dashboard
def dashboard(auth,db,storage,email,password):
    #Marco de datos
    st.title("📊 RESULTADO-DASHBOARD")
    st.markdown("##")
    user = auth.sign_in_with_email_and_password(email,password)
    codigo = db.child(user['localId']).child("Excel").get()
    print(codigo)
    for data in codigo.each():
        dato = data.val()

    df = pd.read_excel(dato)
    #Top KPI
    total_Sembrada=int(df['superficie_sembrada_ha'].sum())
    total_Cosechada =int(df['superficie_cosechada_ha'].sum())
    efect_cosecha=round((int(df['superficie_cosechada_ha'].sum())/int(df['superficie_sembrada_ha'].sum()))*100,1)
    left_column,mid_column,right_column=st.columns(3)
    with left_column:
        st.subheader("Total Sembrada:")
        st.subheader(f"{total_Sembrada:,} t.")
    with mid_column:
        st.subheader("Total Cosecha:")
        st.subheader(f"{total_Cosechada:,} t.")
    with right_column:
        st.subheader("Efectividad:")
        st.subheader(f"{efect_cosecha:,} %")
    st.markdown("---")
    Total_Producción= int(df['produccion_tm'].sum())
    Produccion_Fundo= round(int(df['produccion_tm'].sum())/int(df['superficie_cosechada_ha'].sum()),1)

    Indice_Produccion= df.groupby(by=["fundo_nro"]).sum()[["produccion_tm"]].sort_values(by="fundo_nro").reset_index()
    conteo = df.groupby(by=["fundo_nro"]).sum()[["superficie_cosechada_ha"]].sort_values(by="fundo_nro").reset_index()
    result = round((Indice_Produccion['produccion_tm']/conteo['superficie_cosechada_ha']),2)
    left_column1, mid_column2, right_column3 = st.columns(3)
    with left_column1:
        st.subheader("Total Producción:")
        st.subheader(f"{Total_Producción:,} t.")
    with mid_column2:
        st.subheader("Producción/ha:")
        st.subheader(f"{Produccion_Fundo:,} t.")
    with right_column3:
        st.subheader("Efec. Rendimiento:")
        st.subheader(f"Fundo: {result.idxmax()+1:,}")
        st.subheader(f"Producción: {result.max():,} t.")
    st.markdown("---")
    NFundo=pd.DataFrame(df["fundo_nro"].unique(),columns=["N° Fundo"])
    # NFundo["N° Fundo"].astype(str),
    test = pd.concat([ NFundo["N° Fundo"].astype(str),Indice_Produccion["produccion_tm"], conteo["superficie_cosechada_ha"], result], axis=1)
    test = test.rename(columns={"produccion_tm": "Producción_Total", "superficie_cosechada_ha": "Cosecha_Total",
                         0: "Producción_x_ha"})

    with st.expander("Mostrar dato Estaditico"):
        st.dataframe(test.style.highlight_max(subset=['Producción_Total', 'Cosecha_Total','Producción_x_ha'], color='#76b927').\
                                    highlight_min(subset=['Producción_Total', 'Cosecha_Total','Producción_x_ha'], color='#f53333'), 1000)
    with st.expander("Ingresar Filtro"):
        st.header("Ingrese el filtro:")
        Anio = st.multiselect(
            "Seleccione Año",
            options=df['anio'].unique(),
            default=df['anio'].unique()
        )
        Campaña = st.multiselect(
            "seleccione Campaña",
            options=df['campania'].unique(),
            default=df['campania'].unique()
        )
        NroF = st.multiselect(
            "seleccione N° Fundo",
            options=df['fundo_nro'].unique(),
            default=df['fundo_nro'].unique()
        )
    df_selection= df.query(
        "anio==@Anio & campania==@Campaña & fundo_nro==@NroF"
    )
    if df is not None:
        # df = pd.read_excel("datos_planificacion.xlsx", dtype={'superficie_cosechada_ha': np.float64,
        #                                         'produccion_tm': np.float64,
        #                                         'rendimiento_kgxha': np.float64,
        #                                         'cant_abonoOrg(kg)': np.float64,
        #                                         'abonoTotal(kg)': np.float64},
        #                                         na_values = np.nan)
        with st.expander("Mostrar Datasets"):
            st.dataframe(df_selection.style.format({'superficie_cosechada_ha': '{:.2f}',
                                      'produccion_tm': '{:.2f}',
                                      'rendimiento_kgxha': '{:.2f}',
                                      'cant_abonoOrg(kg)': '{:.0f}','Temp_Semestral': '{:.1f}',
                                      'abonoTotal(kg)': '{:.0f}'}).set_properties(**{'text-align': 'center'}), 1000)


        # SALES BY PRODUCT LINE [BAR CHART]
        f1c1,f1c2=st.columns(2)
        df_selection = df_selection.assign(category="Campaña")
        df_selection['category'] = df_selection['category'].str.cat(df['campania'].astype(str), sep="-")
        Cosecha_by_anio_line = df_selection.groupby(by=["anio"]).sum()[["superficie_cosechada_ha"]].sort_values(by="superficie_cosechada_ha")

        #Grafico Cosecha x Año
        fig_produccion = px.bar(
            Cosecha_by_anio_line,
            x="superficie_cosechada_ha",
            y=Cosecha_by_anio_line.index,
            orientation="h",
            title="<b>Hectarea Cosechada Anualmente</b>",
            color_discrete_sequence=["#0083B8"] * len(Cosecha_by_anio_line),
            template="plotly_white",
        )
        fig_produccion.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=(dict(showgrid=False))
        )

        # Grafico Cosecha x Fundo
        produccion_by_fundo_line = df_selection.groupby(by=["fundo_nro"]).sum()[["superficie_cosechada_ha"]].sort_values(
                                    by="superficie_cosechada_ha")
        fig_produccion_Fundo = px.bar(
            produccion_by_fundo_line,
            y="superficie_cosechada_ha",
            x=produccion_by_fundo_line.index,
            title="<b>Hectarea Cosechada Anualmente</b>",
            color_discrete_sequence=["#0083B8"] * len(produccion_by_fundo_line),
            template="plotly_white",
        )
        fig_produccion_Fundo.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=(dict(showgrid=False)),
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    buttons=[
                        dict(
                            args=["type", "bar"],
                            label="Grafico en Barras",
                            method="restyle"
                        ),
                        dict(
                            args=["type", "pie"],
                            label="Grafico en Círculo",
                            method="restyle",

                        )
                    ],
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.11,
                    xanchor="left",
                    y=1.1,
                    yanchor="top"
                )
            ]
        )
        fig_produccion_Fundo.update_layout(
            annotations=[
                dict(text="Tipo:", x=0.1, xref="paper", y=1, yref="paper",
                     align="right")
            ]
        )
        st.plotly_chart(fig_produccion, use_container_width=True)
        st.plotly_chart(fig_produccion_Fundo, use_container_width=True)

        # Grafico Produccion x Campaña
        grf = df_selection[df_selection["campania"].astype("category").isin([1, 2])].sort_values("anio", ascending=False)
        fig_campaña = px.bar(grf,
                     x="fundo_nro",
                     y="produccion_tm",
                     color="category",
                     color_discrete_sequence=["#18bfae","#df6e03"],
                     text="campania",
                     title="International Visitors",
                     animation_frame="anio",
                     barmode="group"
                        )
        fig_campaña.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=(dict(showgrid=False))
        )
        st.plotly_chart(fig_campaña, use_container_width=True)

        # Grafico Prodiccion x Año y Nro de fundo
        fig_produccion_circular = px.sunburst(
            df_selection,
            values="produccion_tm",
            title="<b>Hectarea Cosechada Anualmente</b>",
            names="anio",
            path=['anio', 'fundo_nro'],


        )
        fig_produccion_circular.update_layout(
            plot_bgcolor="rgba(0,0,0,0)",
            xaxis=(dict(showgrid=False))
        ).update_traces(textinfo="value+label")
        st.plotly_chart(fig_produccion_circular, use_container_width=True)

    else:
        st.error("No hay ningún archivo en el Storage ")

def Manual():
# if pagina == "Manual":
    sidebar()
    #time.sleep(2)
    #Selectores Panel DERECHO
    st.subheader("CONDICIONES ")

    ##extrayendo año actual
    year=str(now.year)
    st.text("Año : " + year )

    #Seleccion de campaña
    campaña= st.radio("Seleccionar Campaña: ",(1, 2))

    #Opciones N° de Fundo
    fundoN=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
    fundopciones = st.selectbox('Seleccione el N° de fundo', fundoN)

    #Superficie Sembrada
    superf_Sembrada = st.number_input('Superficie Sembrada por Hectárea', min_value=0, max_value=50, value=20)

    #Edada de las plantas
    edad_Planta = st.number_input('Edad de la Planta', min_value=0, max_value=14, value=5)
    #Temperatura Promedio
    Temp = st.slider('Temperatura Promedio?', 10.0, 30.0, 20.0)

    # Envia las entradas al modelo.
    if st.button("Predecir"):
        input_arr = np.array([year, campaña, fundopciones, superf_Sembrada,edad_Planta,Temp]).reshape(1, -1)
        resultadoPredicción = prediccion(input_arr, menuopciones)
        st.write('Superficie Cosechada:     ',round(resultadoPredicción[0],2))
def automatico(auth,db,storage,email,password):
# if pagina == "Automático":
    sidebar()
    # Importacion de datos
    uploaded_file = st.file_uploader("Cargue el Archivo", type=['xlsx', 'csv'],
                                             help="Puede seleccionar archivo xlsx/csv para su predicción")
    col1, col2, col3 = st.columns(3)
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
                             highlight_max(subset=['superficie_cosechada_ha','produccion_tm'],color='#76b927').\
                             highlight_min(subset=['superficie_cosechada_ha', 'produccion_tm'], color='#f53333')
                            ,1000)
            datasets['Temp_Semestral'] = round(datasets['Temp_Semestral'], 1)
            left, center, right = st.columns(3)
            descargar= center.button("Descargar Archivo en Excel",key="123")

            if descargar is not None:
                combineData = pd.concat([df, datasets], axis=0)
                # @st.cache
                # def convert_df(df):
                #    return df\
                dato=combineData.to_excel('datos_planificacion.xlsx',index=False)
                # convert_df(combineData)
                user = auth.sign_in_with_email_and_password(email, password)
                path_local = "datos_planificacion.xlsx"
                path_on_cloud = db.child(user['localId']).child("Nombres")\
                                .get().val() + "/Excel/datos_planificacion.xlsx"
                fireb_upload = storage.child(path_on_cloud).put(dato,user['idToken'])
                excel_url=storage.child(path_on_cloud).get_url(fireb_upload['downloadTokens'])
                db.child(user['localId']).child("Excel").push(excel_url)


def pronostico():
    try:
        fecha = str(now.year) + "/" + str(now.month) + "/" + str(now.day)
        hora = str(now.hour) + ":" + str(now.minute) + ":" + str(now.second)
        st.markdown("---")
        cl1, cl2, c3 = st.columns(3)
        cl2.subheader("Fecha: "+fecha)
        cl2.subheader("Hora: "+hora)
        st.markdown("---")
        st.markdown("<header>"
                    "<h1 style='text-align: center'>"+Titulo+"-SENAMHI"+"</h1>"
                    "&nbsp;"
                    "<style type='text/css'>"
                    ".col{width: 33.333%; height: 500px; float: left; text-align: center;padding:0 15px 0 15px;}"
                    "h1,h2,p,header{margin: 10;padding: 5;}"
                    "h3{ display:inline; }"
                    ".col1, .col2, .col3{padding: 3px 10px; border: PowderBlue 2px double;border-top-left-radius: 20px; border-bottom-right-radius: 20px;}"
                    "</style>"
                    "</header>"
                    "<div>"
                    "<div class ='col col1'; >"
                      " <h2 style='text-align: center';>"+ detalle[0]['Fecha'] +"</h2>"
                      "<p style='text-align: center;'>"
                      "<img src="+detalle[0]['Link']+" alt='icono'>"
                      "</p>"
                      "&nbsp; <h3 style='text-align: center; color:#ed3030;'>"+ detalle[0]['Temperatura'].split(sep='/')[0]+"</h3>"
                      "<h3>/</h3>"                                                                                                
                      " <h3 style='text-align: center; color:#2d8ddf;'>"+ detalle[0]['Temperatura'].split(sep='/')[1]+"</h3>"
                    "</br> </br> </br> <p style='text-align: justify;'>"+ detalle[0]['Descripcion'] +"</p>"
                    "</div>"
                    "<div class='col col2'; >"
                     " <h2 style='text-align: center'>"+ detalle[1]['Fecha'] +"</h2>"
                      "<p style='text-align: center;'>"
                      "<img src="+detalle[1]['Link']+" alt='icono'>"
                      "</p>"
                      "&nbsp;  <h3 style='text-align: center; color:#ed3030;'>"+ detalle[1]['Temperatura'].split(sep='/')[0]+"</h3>"
                      "<h3>/</h3>"                                                                                                
                      " <h3 style='text-align: center; color:#2d8ddf;'>"+ detalle[1]['Temperatura'].split(sep='/')[1]+"</h3>"
                    "</br> </br> </br> <p style='text-align: justify;'>"+ detalle[1]['Descripcion'] +"</p>"
                    "</div>"
                    "<div class='col col3';>"
                     " <h2 style='text-align: center'>"+ detalle[2]['Fecha'] +"</h2>"
                     "<p style='text-align: center;'>"
                      "<img src="+detalle[2]['Link']+" alt='icono'>"
                      "</p>"
                      "&nbsp;  <h3 style='text-align: center; color:#ed3030;'>"+ detalle[2]['Temperatura'].split(sep='/')[0]+"</h3>"
                      "<h3>/</h3>"                                                                                                
                      " <h3 style='text-align: center; color:#2d8ddf;'>"+ detalle[2]['Temperatura'].split(sep='/')[1]+"</h3>"
                    "</br> </br> </br> <p style='text-align: justify;'>"+ detalle[2]['Descripcion'] +"</p>"
                    "</div>"                                  
                    "</div>"
                    "&nbsp; "
                    "<div class='row'>"
                    "<div class='table-responsive'>"
                    +str(tabla)+
                    "</div>"
                    "</div>"
                    "&nbsp;"
                    "<p>Fecha: "+str(now.year) + "/" + str(now.month) + "/" + str(now.day)+"</p>"
                    "Fuente: <a href= '"+link+"'>SENAMHI</a>",
                    unsafe_allow_html=True)
    except:
        st.error("La página de Senamhi ha caido")
# if __name__ == "__main__":
#     st_interface()

