import numpy as np

def lowpass_filter(shape, d0=160, ftype='ideal', n=2):
    P, Q = shape
    H = np.zeros((P, Q)) #Inicializa o filtro com zeros

    for u in range(0, P):
        for v in range(0, Q):
            # Obtem a distância euclediana do ponto D(u,v) ao centro
            D_uv = np.sqrt((u - P/2)**2 + (v - Q/2)**2)

            # Define a função de tranferencia Lowpass de acordo com o tipo de filtro
            if ftype == 'ideal':
                if D_uv <= d0:
                    H[u, v] = 1.0
            elif ftype == 'butterworth':
                H[u, v] = 1/(1 + (D_uv/d0)**(2*n))
            elif ftype == 'gaussian':
                H[u, v] = np.exp(-D_uv**2 / (2*d0)**2)
    
    return H

def highpass_filter(shape, d0=160, ftype='ideal', n=2):
    # Inverso do filtro lowpass
    H = 1.0 - lowpass_filter(shape, d0, ftype, n)
    return H

def bandreject_filter(shape, d0=160, w=20, ftype='ideal', n=2):
    P, Q = shape
    H = np.ones((P, Q))

    for u in range(0, P):
        for v in range(0, Q):
            # Obtem a distância euclediana do ponto D(u,v) ao centro
            D_uv = np.sqrt((u - P/2)**2 + (v - Q/2)**2)

            # Define a função de tranferencia Lowpass de acordo com o tipo de filtro
            if ftype == 'ideal':
                if (d0 - (w/2)) <= D_uv <= (d0 + (w/2)):
                    H[u, v] = 0.0
            elif ftype == 'butterworth':
                if D_uv == d0:
                    H[u, v] = 0.0
                else:
                    H[u, v] = 1/(1 + ((D_uv*w)/(D_uv**2 - d0**2))**(2*n))
            elif ftype == 'gaussian':
                if D_uv == 0:
                    H[u, v] = 1.0
                else:
                    H[u, v] = 1.0 - np.exp(-((D_uv**2 - d0**2)/(D_uv*w))**2)
    
    return H

def bandpass_filter(shape, d0=160, w=20, ftype='ideal', n=2):
    # Inverso do rejeita banda
    H = 1.0 - bandreject_filter(shape, d0, w, ftype, n)
    return H

def filter_image_freq(img, fclass='lowpass', ftype='ideal', d0=160, w=20, n=2):
    M, N = img.shape
    F = np.fft.fft2(img.astype(np.float32))
    F = np.fft.fftshift(F)
    H = np.zeros((M, N))
    if fclass == 'lowpass':
        H = lowpass_filter(F.shape, d0, ftype, n)
    elif fclass == 'highpass':
        H = highpass_filter(F.shape, d0, ftype, n)
    elif fclass == 'bandreject':
        H = bandreject_filter(F.shape, d0, w, ftype, n)
    elif fclass == 'bandpass':
        H = bandpass_filter(F.shape, d0, w, ftype, n)
    
    G = F * H
    G = np.fft.ifftshift(G)
    G = np.fft.ifft2(G)
    G = np.abs(G)

    return G
