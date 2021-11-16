from bs4 import BeautifulSoup
import re
import requests


def webScraping():
    global Titulo, detalle
    link="https://www.senamhi.gob.pe/?p=pronostico-detalle&dp=13&localidad=0005"
    senamhi = requests.request('GET',link)
    soup = BeautifulSoup(senamhi.text)
    characters = "\n"

    #subtitulo
    p=soup.find('h2', attrs={'class':'subtit-interior'})
    Titulo=p.text

    #Detalle Diario (Fecha, Temperatura, Descripción)
    p1=soup.find('article', attrs={'class':'prono-detalle-hor'})
    caja=p1.find_all('div',attrs={'class':'col-lg-4'})

    detalle = list()
    count=0
    for i in caja:
     if count < 3:
       fecha=re.sub(characters, "",i.find('p').get_text())
       link_IMG = "https://www.senamhi.gob.pe/" + i.find('img').get('src')
       temp=re.sub(characters, "",i.find('h4').get_text())
       dec=re.sub(characters, "",i.find('p', attrs={'class':'desc'}).get_text())
       detalle.append({'Fecha': fecha,'Link':link_IMG,'Temperatura':temp,'Descripcion':dec})
     else:
      break
     count +=1

    p4 = soup.find('div', attrs={'class': 'tab-pane fade'})
    tabla = p4.find('table')
    return Titulo, detalle,tabla,link

webScraping()