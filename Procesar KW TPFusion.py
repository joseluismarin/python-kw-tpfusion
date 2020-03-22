#!/usr/bin/env python
# coding: utf-8
# Author: Jlmarin
# Web: https://jlmarin.eu
# Version: 1.0.0

import argparse
import sys
import pandas as pd
from nltk import SnowballStemmer
import spacy
import es_core_news_sm
from tqdm import tqdm
from unidecode import unidecode
import glob
import re
import urllib.parse


parser = argparse.ArgumentParser()
parser.add_argument('-f', '--file', help='Nombre del archivo a procesar')
parser.add_argument('-s', '--save', help='Nombre del archivo al guardar')
parser.add_argument('-i', '--intent', nargs='?', const=1, type=bool, default=True, help='Activa el procesado de las intenciones de busqueda')
parser.add_argument('-l', '--location', nargs='?', const=1, type=bool, default=True, help='Nombre del archivo con la base de datos de las localizaciones')
parser.add_argument('-d', '--debug', help='Limita los bucles al numero de operaciones indicadas')
args = parser.parse_args()

pd.options.mode.chained_assignment = None 

nlp = es_core_news_sm.load()
spanishstemmer=SnowballStemmer('spanish')

def normalize(text):
    text = unidecode(str(text))
    doc = nlp(text)
    words = [t.orth_ for t in doc if not t.is_punct | t.is_stop]
    lexical_tokens = [t.lower() for t in words if len(t) > 3 and t.isalpha()]
    
    return lexical_tokens

def raiz(kw):
    #Calculamos la raiz semantica
    stems = [spanishstemmer.stem(wd) for wd in kw]
    raiz = " ".join(sorted(stems))
    return raiz


if not args.file:
    print('Error: Argumento -f no encontrado.')
    sys.exit()

# Abrimos el archivo y los agregamos a un dataframe  
df = pd.read_csv(args.file,sep='\t')
print('Archivo cargado... OK')

# Calculamos raiz semantica
loop = tqdm(total = len(df.index), position = 0, leave = False)

df['Raiz semantica'] = ''

for i in df.index:
    loop.set_description("Calculando raices...".format(i))

    kw_a = normalize(df.loc[i,'Keyword'])

    #Calculamos la raiz semantica
    df.loc[i,'Raiz semantica'] = raiz(kw_a)

    loop.update(1)
    if args.debug and i>int(args.debug):
        break
loop.close()
print('Calculado raices semanticas... OK')
df = df.sort_values(by=['Raiz semantica', 'Volume'], ascending=[True,False])
df = df.reset_index(drop=True)

# Agrupamos las keywords segun su raiz semantica y el volumen de busquedas
loop = tqdm(total = len(df.index), position = 0, leave = False)

df['Grupo'] = ''
for i in df.index:
    loop.set_description("Agrupando...".format(i))
    if i == 0:
        df.loc[i,'Grupo'] = df.loc[i,'Keyword']
    elif df.loc[i,'Raiz semantica'] == df.loc[i-1,'Raiz semantica']:
        df.loc[i,'Grupo'] = df.loc[i-1,'Grupo']
    else:
        df.loc[i,'Grupo'] = df.loc[i,'Keyword']
        
    loop.update(1)
    
loop.close()
print('Agrupado... OK')

#Guardamos el archivo procesado
df.to_csv('kw_procesado.csv', index=False)
print('Archivo kw_procesado.csv creado... OK')

#Renombro nombre columna para evitar error al agrupar
df.rename(columns={'Pos.MedRivales':'PosMedRivales'})

gdf = (df.groupby('Grupo', as_index=False)
    .agg({
        'Position':'mean',
        'Volume':'sum',
        'Difficulty':'mean',
        'CPC':'mean',
        'Pos.Med\nRivales': 'mean',
        'Keyword':' | '.join
    }))

def componer_URL(texto):
    return f'https://www.google.es/search?source=hp&q={urllib.parse.quote(texto)}&pws=0'

gdf['Check SERPs'] = [componer_URL(h) for h in gdf['Grupo']]


# Detectamos la intencion de busqueda de la kw: Informacional, transacional, navegacional
if args.intent:
    intenciones = pd.read_csv('Data/intenciones.csv')

    loop = tqdm(total = len(intenciones.index), position = 0, leave = False)

    gdf['Intencion'] = ''
    for i in intenciones.index:
        loop.set_description("Detectando intenciones de busqueda...".format(i))

        row = gdf[gdf['Grupo'].str.contains(rf'\b{str(intenciones.loc[i,"Patron"])}\b')]

        if row is not None:
            gdf.loc[row.index,'Intencion'] = intenciones.loc[i,'Tipo']
        
        loop.update(1)

    loop.close()
    print('Intenciones de busqueda... OK')

# Detectamos la ubicacion de la palabra clave.
if args.location:
    ubicaciones = pd.read_csv('Data/ubicaciones.csv')

    loop = tqdm(total = len(ubicaciones.index), position = 0, leave = False)

    gdf['Ubicacion'] = ''
    gdf['Tipo ubicacion'] = ''
    for i in ubicaciones.index:
        loop.set_description("Detectando ubicaciones...".format(i))

        row = gdf[gdf['Grupo'].str.contains(rf'\b{str(ubicaciones.loc[i,"Ubicacion"])}\b')]

        if row is not None:
            gdf.loc[row.index,'Ubicacion'] = ubicaciones.loc[i,'Ubicacion']
            gdf.loc[row.index,'Tipo ubicacion'] = ubicaciones.loc[i,'Tipo']
        
        loop.update(1)

    loop.close()
    print('Ubicaciones... OK')

gdf.to_csv('kw_agrupado.csv',index=False)
print('Archivo kw_agrupado.csv creado... OK')
print('Proceso finalizado... OK')      

"""
file_to_save = 'SALIDA_PROCESADA.csv'
if args.save:
    file_to_save = args.save

df.to_csv(file_to_save, index=False)
print('Proceso finalizado') """



