# -*- coding: utf-8 -*-
"""
Created on Mon May 10 13:08:06 2021

@author: Diego Morales
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import skimage.io as io
from scipy import fftpack
import random

#Leemos la imagen seleccionada; la guardamos en f
f=io.imread('RealMadrid.png')

#Dimensiones de la imagen
Nx=len(f)
Ny=len(f[0])

#Escala de grises
fGris=np.zeros((Nx,Ny))
for i in range(Nx):
    for j in range(Ny):
        fgris=(0.2125*f[i][j][0]+0.7154*f[i][j][1]+0.0721*f[i][j][2])/255
        fGris[i][j]=fgris
#Mostramos la imagen en escala de grises
print('Imagen original')
plt.imshow(fGris,cmap='gray')
plt.savefig('RealMadrid Gris.jpeg')
plt.show()

#Empezamos con la matriz para la DCT
piece=np.zeros((8,8))
CosDis=np.zeros((Nx,Ny))
#Variables para transformada
startx=0
countx=0
starty=0
county=0
size=8

#Transformada Coseno discreta
while county<80:
    starty=8*county
    endy=starty+size
    countx=0
    while countx<53:
        startx=8*countx
        endx=startx+size
        #Partición en pedazos de 8x8 pixeles
        for i in range(0,size):
            for j in range(0,size):
                if(i+startx)<427:
                    if(j+starty)<640:
                        piece[i,j]=fGris[i+startx,j+starty]
        #Transformada Coseno Discreta del pedazo elegido
        dctpiece=fftpack.dct(piece)
        #Agregamos el trozo a la matriz completa
        for i in range(0,size):
            for j in range(0,size):
                if(i+startx)<427:
                    if(j+starty)<640:
                        CosDis[i+startx,j+starty]=dctpiece[i,j]
            countx+=1
        county+=1
#Evaluación para mostrar una parte de la imagen transformada
valx=random.randint(0,79)
valy=random.randint(0,52)
for i in range(0,size):
    for j in range(0,size):
        if(i+8*valx)<430:
            if(j+8*valy)<645:
                piece[i,j]=fGris[i+8*valx,j+8*valy]
#Transformada de la parte a mostrar
dctpiece=fftpack.dct(piece)

print('Ubicación de porción de imagen',8*valy,8*valx)
print('Porción de imagen original')
plt.imshow(piece,cmap='gray')
plt.show()
print('Porción de imagen transformada')
plt.imshow(dctpiece,cmap='gray')
plt.show()

#Coeficientes del umbral

total=0
NoCero=0
umbral=0.012
for i in range(Nx):
    for j in range(Ny):
        if(CosDis[i,j]>umbral):
            CosDis[i,j]=CosDis[i,j]*umbral
            NoCero+=1
        else:
            CosDis[i,j]=0
            total+=1
porcentaje=100*NoCero/total
print('Porcentaje del tamaño final de la imagen ',porcentaje,'%')
#Inicialización de matrices para transformada coseno discreta
idctf=np.zeros((Nx,Ny))
#Inicialización de variables para TCD inversa
startx=0
countx=0
starty=0
county=0
#Transformación Inversa Coseno Discreta
while county<80:
    stary=8*county
    endy=starty+size
    countx=0
    while countx<53:
        startx=8*countx
        endx=startx+size
        #Partición en pedazos de 8x8 pixeles
        for i in range(0,size):
            for j in range(0,size):
                if(i+startx)<480:
                    if(j+stary)<645:
                        piece[i,j]=CosDis[i+startx,j+starty]
        #Transformada inversa coseno discreta del pedazo en cuestión
        idctpiece=fftpack.idct(piece)
      #Agregar el pedazo transformado a la matriz completa
        for i in range(0,size):
           for j in range(0,size):
               if(i+startx)<427:
                   if(j+starty)<640:
                       idctf[i+startx,j+startx]=idctpiece[i,j]
           countx+=1
        county+=1

print('Imagen Original gris')                    
plt.imshow(fGris,cmap='gray')
plt.show()
print('Imagen Comprimida')
plt.imshow(idctf,cmap='gray')
plt.savefig('Inverse Cosine Transformed RealMadrid.jpeg')
plt.show()

#Transformada de Fourier

#Inicialización de matrices para Transformada de Fourier
piece=np.zeros((8,8))
ftf=np.zeros((Nx,Ny))
#Inicialización de variables para TCD
startx=0
countx=0
starty=0
county=0

#Transformada Coseno Discreta
while county<80:
    starty=8*county
    endy=starty+size
    countx=0
    while countx<53:
        startx=8*countx
        endx=startx+size
        
        for i in range(0,size):
            for j in range(0,size):
                if(i+startx)<427:
                    if(j+starty)<640:
                        piece[i,j]=fGris[i+startx,j+starty]
        #Transformada Coseno Discreta del pedazo en cuestión
        ftpiece=fftpack.fft2(piece)
        #Agregar el pedazo transformado a la matriz completa
        for i in range(0,size):
            for j in range(0,size):
                if(i+startx)<427:
                    if(j+starty)<640:
                        ftf[i+startx,j+starty]=ftpiece[i,j]
            countx+=1
        county+=1

#Evaluación para mostrar una parte de la imagen transformada
valxx=random.randint(0,79)
valy=random.randint(0,52)
for i in range(0,size):
    for j in range(0,size):
        if(i+8*valx)<430:
            if(j+8*valy)<645:
                piece[i,j]=fGris[i+8*valx,j+8*valy]
#Transformada de la parte a mostrar
ftpiece=fftpack.fft2(piece) 
#Mostramos la imagen transformada
print('Imagen transformada')
plt.imshow(ftf.real,cmap='gray')
plt.savefig('RealMadrid Transformada de Fourier.jpeg')
plt.show()

#Coeficientes con el umbral  
total=0
NoCero=0
for i in range(Nx):
    for j in range(Ny):
        if(ftf[i,j])>umbral:
            ftf[i,j]=ftf[i,j]*umbral
            NoCero+=1
        else:
            ftf[i,j]=0
        total+=1
porcentaje=100*NoCero/total
print('Porcentaje del tamaño final de la imagen',porcentaje,'%')
#Inicialización de matrices para transformada coseno discreta
iftf=np.zeros((Nx,Ny))
#Inicialización de variables para transformada inversa coseno discreta
startx=0
countx=0
starty=0
county=0

#Transformada inversa coseno discreta
while county<80:
    starty=8*county
    endy=starty+size
    countx=0
    while countx<53:
        startx=8*countx
        #Participación en pedazos de 8x8 pixeles
    for i in range(0,size):
        for j in range(0,size):
            if(i+startx)<430:
                if(j+starty)<645:
                    piece[i,j]=ftf[i+startx,j+starty]
    #Transformada inversa coseno discreta del pedazo en cuestión
    iftpiece=fftpack.ifft2(piece)
#Agregar el pedazo detransformado a la matriz completa
for i in range(0,size):
    for j in range(0,size):
        if(i+startx)<427:
            if(j+starty)<640:
                iftf[i+startx,j+starty]=iftpiece[i,j]
    countx+=1
county=1

print('Imagen original gris')
plt.imshow(fGris,cmap='gray')
plt.show()
print('Imagen comprimida')
plt.imshow(iftf,cmap='gray')
plt.savefig('RealMadrid Transformada Inversa Fourier.jpeg')
plt.show()
    
           






