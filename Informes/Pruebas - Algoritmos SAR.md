# Pruebas - Algoritmos SAR

1. **Formar la imagen SAR del siguiente objetivo simulao, el cual está conformado por 3 blancos puntuales:**

![Untitled](Pruebas%20-%20Algoritmos%20SAR/Untitled.png)

A continuación, definimos la siguiente función para generar los 3 blancos puntuales.. Se considera que cada blanco puntual tiene magnitud 1 y fase 0 rad.

```python
def get_scalar_data3():

    at=np.array([1,1,1]) # Reflectividad
    Rt=np.array([(-2,2),(0,4),(2,6)]) # Coordenadas de los 3 blancos puntuales (x,y)m

    return at,Rt
```

Asimismo, se deben utilizar los siguientes parámetros:

| Parámetro | Valor |
| --- | --- |
| Frecuencia central, ⁍ | 15 GHz |
| Ancho de banda ⁍ | 600 MHz |
| Número de frecuencias ⁍  | 41 |
| Apertura sintética ⁍ | 0.3 m |
| Ángulo de visión ⁍ | 90 ° |
| Número de posiciones ⁍ | 62 |
| Ancho de la imagen ⁍ | 8m |
| Ancho de la imagen ⁍ | 8m |
| Grilla en azimut ⁍ | 1.25 cm |
| Grilla en rango ⁍ | 1.25 cm |

A continuación, se define una nueva función para definir los nuevos parámetros:

```python
# Funcion que define los parametros del sistema (Ejercicio de Simulacion)
def get_parameters_sim():
    # Definición de parámetros
    c = 0.3 #0.299792458 # Velocidad de la luz (x 1e9 m/s)
    fc = 15 # Frecuencia Central(GHz)
    BW = 0.6 # Ancho de banda(GHz)
    Ls = 0.3 # 4 # 0.6 # Longitud del riel (m)
    Ro = 0 # Constante debido al delay de los cables(m)
    theta = 90 # Angulo azimuth de vision de la imagen final(grados sexagesimales E [0-90])

    # Definicion de las dimensiones de la imagen ("dh" y "dw" usados solo en BP)
    w = 10 # 800 # 5 # Ancho de la imagen(m)
    h = 10 #10 # 800 # 5 # Altura de la imagen(m)
    dw = 1.25 #0.25 # 0.2 #0.1 # Grilla en azimut(m)
    dh = 1.25 #0.25 # 0.2 #0.1 # Grilla en rango(m)

    # Hallando el Np a partir de los pasos
    dp= c/(4.21*fc*np.sin(theta*np.pi/180)) # paso del riel para un angulo de vision de 180°
    Np= int(Ls/dp)+1 # Numero de pasos del riel = 62

    if Np%2!=0:
        Np+=1   # Para que el numero de pasos sea par

    # Hallando el Nf en funcion a la distancia máxima deseada
    r_r=c/(2*BW) # resolucion en rango
    Nf=int(h/r_r) +1 #Numero de frecuencias = 41

    prm={
        'c':c,
        'fc':fc,
        'BW':BW,
        'Ls':Ls,
        'Ro':Ro,
        'theta':theta,
        'Np':Np,
        'Nf':Nf,
        'w':w,
        'h':h,
        'dw':dw,
        'dh':dh
    }
    return prm
```

## Algoritmo FDBP y RMA

1. **Formación de la imagen SAR en base a los parámetros anteriores:**
    
    **Gráficas de Magnitud**
    
    ![RD1_2)prueba_1_mag.png](Pruebas%20-%20Algoritmos%20SAR/RD1_prueba_1_mag.png)
    
    Como se puede observar en la gráfica de magnitud, el algoritmo FDBP logra reconstruir de forma adecuada los blancos. Sin embargo, se nota de forma pronunciada los lóbulos laterales, pues superan los -100 dB. Asimismo, debido a la resolción en azimut y rango son mucho mayores que la resolución de la grilla (0.25 y 0.0667 > 0.0125), los blancos son detectados de forma más intensa. Por otro lado, el algoritmo RMA logra reconstruir de forma adecuada los dos objetivos más lejanos. No obstante, el objetivo más cercano no se distingue dde forma adecuada, esto es debido a que el algoritmo RMA no reconstruye blancos menores a un cierto ángulo.
    
    **Gráficas de Fase**
    
    ![RD1_prueba_1_phase.png](Pruebas%20-%20Algoritmos%20SAR/RD1_prueba_1_phase.png)
    
    En las gráficas de fase se puede observar que el algortimo FDBP presenta una fase constante, mientras que el algoritmo RMA presenta pequeñas variaciones a lo largo de toda la fase.
    
2. **Calcular la resolución en rango y azimut de la imagen SAR. Indicarlo en la imagen**
    
    **La resolución en rango** está determinado por la habilidad del radar de distinguir 2 blancos puntuales por separado en la dirección del rango inclinado. Y se calcula como sigue:
    
    $\Delta r=\frac{c}{2BW} = 0.25 m$ 
    
    La **resolución en azimuth** está determinado por la habilidad del radar de distinguir 2 blancos puntuales por separado en la dirección del azimuth. Y se calcula como sigue:
    
    $\Delta x = r \frac{\lambda}{2L_s}=r \frac{c}{2f_cL_s} = \frac{r}{30}$
    
    Como vemos en la ecuación anterior, **el rango en azimuth** depende directamente del rango al cuál se encuentre el objetivo. A continuación, calculamos el rango en azimuth para cada uno de los 3 objetivos.
    
    - Primer objetivo ((-2, 2) → r = 2): $\Delta x = 0.0667 m$
    - Segundo objetivo ((0, 4)→ r = 4): $\Delta x = 0.1333 m$
    - Tercer objetivo ((2, 6) → r = 6): $\Delta x = 0.2 m$
    
    Asimismo, verificamos el valor del **rango crítico**; $R_c = \frac{L_s}{\theta}=0.19m$. Esto indica que para distancias menores a 0.19m, la resolución en azimut es igual a $\Delta x = L_s = 0.3m$. Sin embargo, ninguno de los objetivos cae dentro de este rango. Asimismo, el **rango máximo permitido** se calcula como $R_u = N_f*\Delta r = 10.25 m$, por lo que el parámetro es satisfecho dado que el rango del objetivo más lejano es 6 m.
    
3. **Formar la imagen SAR, con la mitad de resolución en rango y azimut que en el caso anterior.**
    
    Para disminuir la resolución a la mitad se decidió incrementar en un factor de 2 el ancho de banda (para la resolución en rango) y a la apertura sintética (para la resolución en azimut). Para este último se tiene en cuenta que al aumentar el valor de la apertura sintética, también se aumenta el valor de $N_p$. Asimismo, se varía el valor de $N_f$ en base a BW, de lo contrario el rango máximo sería 5.125 y no se podría recuperar el objeto más lejano.
    
    **Gráfico de Magnitud**
    
    ![RD1_2)prueba_2_mag.png](Pruebas%20-%20Algoritmos%20SAR/RD1_prueba_2_mag.png)
    
    Los objetivos se detectan de manera simular que en el caso previo. La principal diferencia es que los blancos tienden a ser más puntuales, esto debido a que la resolución en ambos ejes disminuyó.
    
    **Gráfico de Fase**
    
    ![RD1_2)prueba_2_phase.png](Pruebas%20-%20Algoritmos%20SAR%20c5432bbbdb8b4ce4af019e233f2e3bc7/RD1_2)prueba_2_phase.png)
    
    La fase obtenida con el algoritmo FDBP tiende a ser constante, mientras que la obtenida por el algoritmo RMA presenta variaciones.
    
4. **Incrementar la altura de la imagen $L_y = 15$ y formar la imagen. Comentar los resultados**
    
    Al incrementar el valor de la atura de la imagen se modifica indirectamente el rango máximo permitido. Esto debido a que se varía el valor de $N_f$.
    
    Esto se refleja en las dimensiones de la imagen, sin embargo, esto no afecta en la detección de los objetivos.
    
    **Gráfico de Magnitud**
    
    ![RD1_prueba_3_mag.png](Pruebas%20-%20Algoritmos%20SAR/RD1_prueba_3_mag.png)
    
    **Gráfico de Fase**
    
5. **Si cambiamos la escena de la imagen rectangular a triangular, formar la imagen SAR empleando el algoritmo FDBP, y los parámetros de la tabla anterior. ¿Se podrá emplear el algoritmo RMA?**
    
    **Gráfico de Magnitud**
    
    ![RD1_prueba_4_mag.png](Pruebas%20-%20Algoritmos%20SAR/RD1_prueba_4_mag.png)
    
    **Gráfico de Fase**
    
    ![RD1_2)prueba_4_phase.png](Pruebas%20-%20Algoritmos%20SAR/RD1_prueba_4_phase.png)