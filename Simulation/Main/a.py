import numpy as np
import scipy.interpolate as sc
c = 1; Ro = 1; Np =1; Nf = 1;
Ls = 1; dy = 1; Lx =1 ; Ly = 1; dx =1 

# Funcion para calcular la distancia entre una posición del riel y todos las coordenadas de la imagen
def distance_nk(r_n,x_k): # vector de coordenadas de la imagen, punto del riel "k"
    alpha = t_sq*np.pi/180
    R = np.array([[np.cos(alpha), np.sin(alpha)],[-np.sin(alpha), np.cos(alpha)]])
    r_n = R.dot(r_n.T)
    r_n = r_n.T
    d=((r_n[:,0]-x_k*np.cos(alpha))**2+(r_n[:,1])**2)**0.5
    return d

    
def FDBP_Algorithm(data1):
    """ Ejecuta el algoritmo Frequency Domain Back Projection"""
    # Lectura de parametros
    Ski = data1['Ski'].copy()
    df = data1['df']; fi = data1['fi']; rr_r = data1['rr_r']
    Lista_pos = np.linspace(-Ls/2, Ls/2, Np)

    # Blackman Windows
    S1 = Ski*np.blackman(len(Ski[0])) # Multiply rows x hamming windows
    S1 = (S1.T*np.blackman(len(S1))).T # Multiply columns x hamming windows    
    
    #----------PRIMER PASO: 1D-IFFT respecto al eje de la 'f'-----------
    # a) Agregar un factor de fase debido al delay de los cables. Ro = 0, entonces no tiene efecto en este caso.
    S2 = S1*np.vander([np.exp(-1j*4*np.pi/c*Ro*df)]*Np,Nf,increasing=True) # Vandermonde matrix
    
    # b) Efectuar un zero padding en el eje de la 'f'
    zpad = 3*int(rr_r*(Nf-1)/dy)+1 # Dimension final despues del zero padding 
    col = int(zpad - len(S1[0])) # Length of zeros
    if col>=0:
        S3 = np.pad(S2, [[0, 0], [0, col]], 'constant', constant_values=0) # Aplica zero padding a ambos extremos de la matriz
    else:
        S3 = S2[:,:zpad]

    # Funcion de interpolacion
    def interp(k,dist): # k: posicion del riel, vector de distancias entre 'n' y 'k'
        fr=sc.interp1d(rn,S4[k].real,kind='linear',bounds_error=False, fill_value=S4[k,-1].real) # Interpolacion al punto mas cercano
        fi=sc.interp1d(rn,S4[k].imag,kind='linear',bounds_error=False, fill_value=S4[k,-1].imag) # Fuera de la frontera se completa con el valor de la frontera
        return fr(dist) +1j*fi(dist)

    # c) Compresión en Rango: Efectuar la IFFT
    S4 = np.fft.ifft(S3,axis=1) # A lo largo de la dimensión de frecuencia

    #------------------SEGUNDO PASO: Interpolacion-----------------------
    dkn = 4*np.pi*df/c
    rn = (np.arange(zpad))*2*np.pi/dkn/(zpad-1)
    x_c=np.arange(-Lx/2,Lx/2+dx,dx); y_c=np.arange(0,Ly+dy,dy)
    r_c=np.array([(i,j) for j in y_c for i in x_c])

    #------------------TERCER PASO: SUMA COHERENTE-----------------------
    S5=np.zeros(len(r_c),dtype=complex)
    for kr in range(Np):
        Rnk = distance_nk(r_c,Lista_pos[kr]) # Vector de distancias entre una posicion del riel y todos los puntos de la imagen
        Ke = np.exp(1j*4*np.pi/c*fi*(Rnk-Ro)) # Factor de fase
        Fnk = interp(kr,Rnk) # Valor interpolado en cada punto "n" de la grilla
        S5 += Fnk*Ke # Valor final en cada punto de la grilla
    S6 = S5/(Nf*Np)
    S7 = np.reshape(S6,(len(y_c),len(x_c)))

    #------------------CUARTO PASO: NORMALIZACION-----------------------
    Im = S7/(np.sqrt(np.sum(abs(S7)**2)))

    return {'Im':Im}