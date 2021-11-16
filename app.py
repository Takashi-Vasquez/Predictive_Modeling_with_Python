import pyrebase
import streamlit as st
import time
from datetime import datetime
import view
from PIL import Image

#configuracion de la pagina
st.set_page_config(page_title="HassPerú", page_icon="🥑")
hide_st_style = """
            <style>
            --#MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            --header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# Configuracion de  clave
firebaseConfig = {
  'apiKey': "AIzaSyDmoxXKMrMzBpnZZ4l6e34L_fMTZ6LoqJI",
  'authDomain': "db-hassperu.firebaseapp.com",
  'projectId': "db-hassperu",
  'databaseURL': "https://db-hassperu-default-rtdb.firebaseio.com/",
  'storageBucket': "db-hassperu.appspot.com",
  'messagingSenderId': "25961481823",
  'appId': "1:25961481823:web:9aceae0812ff253fa83dea",
  'measurementId': "G-6PXM72T6PC"
}
#variable global
global auth,db,storage,email,password

# Firebase Authentication
firebase = pyrebase.initialize_app(firebaseConfig )
auth = firebase.auth()

# Database
db = firebase.database()
storage = firebase.storage()


# Autentificación
choice = st.sidebar.selectbox('login/Signup', ['Login', 'Sign up'])

# Obtain User Input for email and password
email = st.sidebar.text_input('Correo')
password = st.sidebar.text_input('Contraseña',type = 'password')
roles = {"B*WAjkMR*:": "Owner", "iM8IA3": "Manager", "hpHIKY": "Operator"}

# Sign up Block

if choice == 'Sign up':
  Nombres = st.sidebar.text_input('Nombre y Apellido')
  usuario = st.sidebar.text_input('Nombre de usuario')
  number = st.sidebar.text_input('Número de celular',max_chars=9)

  # asignacion de roles
  clave = st.sidebar.text_input('Clave de credencial')
  submit = st.sidebar.button('Crear Cuenta')

  if submit:
    if clave in roles:
      user = auth.create_user_with_email_and_password(email,password)
      st.success('Cuenta creada exitosamente')
      st.balloons()
      # Sign in
      user = auth.sign_in_with_email_and_password(email, password)
      db.child(user['localId']).child("Nombres").set(Nombres)
      db.child(user['localId']).child("User").set(usuario)
      db.child(user['localId']).child("PhoneNumber").set(number)
      db.child(user['localId']).child("Rol").set(roles[clave])
      db.child(user['localId']).child("ID").set(user['localId'])
      st.title('Welcome' + usuario)
      st.info('Login via login drop down selection')
    else:
      st.error("Error al registrar")

  else :
    st.title("Los mejores frutos de nuestra tierra by HassPerú")
    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
    imagen = Image.open("Image/HassPeru.png")
    st.image(imagen, caption='Frescura, calidad y sabor', width=700)
    st.markdown("<p style='text-align: left; color: #789d10;'>Hass Perú es una empresa agroindustrial con más de "
                "10 años de experiencia en el sector, dedicada al cultivo y comercialización de"
                "paltas Hass y arándano a nivel nacional e internacional.</p>", unsafe_allow_html=True)
    st.markdown("Web Site: [HASS PERÚ](http://www.hassperu.com/es/)")

# Login Block
elif choice == 'Login':
  st.write('<style type="text/css">.css-m96wse {border: none;outline:none;background-color: transparent;}</style>', unsafe_allow_html=True)
  check = st.sidebar.button('¿Olvidaste la contraseña?', key='Check')
  if check:
    with st.spinner(text="Se envio un correo"):
      time.sleep(3)
      try:
        auth.send_password_reset_email(email)
      except:
        st.error("¡Ocurrio un error intenta más tarde!")

  checkbox = st.sidebar.checkbox('Ingresar')
  if checkbox:
    try:
      user = auth.sign_in_with_email_and_password(email, password)
      st.subheader("BIENVENIDO " + db.child(user['localId']).child("User").get().val().upper())
      st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
      clasificador=db.child(user['localId']).child("Rol").get().val()
      if clasificador == "Owner":
        # MENU BAR
        pagina = st.radio('Cambiar Página', ['🥑Home', '📊Dashboard', '💪Manual', '🦾Automático','🌤️Pronostico'])
        if pagina[1::]=="Home":
          view.home()
        elif pagina[1::]=="Dashboard":
          view.dashboard(auth,db,storage,email,password)
        elif pagina[1::]=="Manual":
          view.Manual()
        elif pagina[1::]=="Automático":
          view.automatico(auth,db,storage,email,password)
        elif pagina[2::]=="Pronostico":
          view.pronostico()

      elif clasificador == "Manager":
        # MENU BAR
        pagina = st.radio('Cambiar Página', ['Home','Reporte', 'Manual', 'Automático'])
        if pagina == "Home":
          view.home()
        elif pagina == "Manual":
          view.Manual()
        elif pagina == "Automático":
          view.automatico(auth,db,storage,email,password)
      elif clasificador == "Operator":
        # MENU BAR
        pagina = st.radio('Cambiar Página', ['Home', 'Manual', 'Automático'])
        if pagina == "Home":
          view.home()
        elif pagina == "Manual":
          view.Manual()
        elif pagina == "Automático":
          view.automatico(auth,db,storage,email,password)

    except :
        st.error("Contraseña incorrecta")


  else :
    st.title("Los mejores frutos de nuestra tierra by HassPerú")
    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)
    imagen = Image.open("Image/HassPeru.png")
    st.image(imagen, caption='Frescura, calidad y sabor', width=700)
    st.markdown("<p style='text-align: left; color: #789d10;'>Hass Perú es una empresa agroindustrial con más de "
                "10 años de experiencia en el sector, dedicada al cultivo y comercialización de"
                "paltas Hass y arándano a nivel nacional e internacional.</p>", unsafe_allow_html=True)
    st.markdown("Web Site: [HASS PERÚ](http://www.hassperu.com/es/)")






