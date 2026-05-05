import numpy as np
import math

# ==========================================
# PARÁMETROS DEL SUSTRATO Y FRECUENCIA
# ==========================================
f = 2e9               # Frecuencia de trabajo: 2 GHz
c = 3e8               # Velocidad de la luz en el vacío (m/s)

# Sustrato
er = 4                # Constante dieléctrica relativa
H = 1.57e-3           # Altura del sustrato en metros (1.57 mm)
# t = (1.625e-3 - H)/2 (siendo 1.625e-3 m el espesor total de la placa)
t = 27.5e-6           # Espesor de la pista de cobre en metros (27.5 um)

# Parámetros a calcular: Transformadores L/4
#Z_QWin  = 22.13       # Impedancia característica L/4 entrada (de 9.79 a 50 Ohm)
#Z_QWin  = 31.29       # Impedancia característica L/4 entrada (de 9.79 a 100 Ohm)
#Z_QWin  = 70.71       # Impedancia característica L/4 entrada (de 100 a 50 Ohm)
#Z_QWin  = 27.097       # Impedancia característica L/4 entrada (de 9.79 a 75 Ohm)
#Z_QWin  = 61.237       # Impedancia característica L/4 entrada (de 75 a 50 Ohm)
#Z_QWin  = 30       # Impedancia característica L/4 "capacitor" de polarizacion de base (circuito abierto a cortocircuito)
Z_QWin  = 90       # Impedancia característica L/4 "inductor" de polarizacion de base (cortocircuito a circuito abierto)

Z_QWout = 36.78       # Impedancia característica L/4 salida (de 27.05 a 50 Ohm)

# Parámetros a calcular: Capacitores de compensación
C_in    = 0.920e-12   # Capacitor de entrada (F) -> 0.920 pF
C_out   = 3.545e-12   # Capacitor de salida (F) -> 3.545 pF

# Impedancias características de las microtiras para los capacitores
Z0_Comp_in  = 75      # Impedancia del stub para C_in (Ohm)
Z0_Comp_out = 50      # Impedancia del stub para C_out (Ohm)

# ==========================================
# FUNCIONES DE HAMMERSTAD Y WHEELER
# ==========================================
def calcular_geometria_microtira(Z0, er, H, t):
    """
    Calcula W, We, E_re, W/H y comprueba Z0 usando Hammerstad y Wheeler simultáneamente.
    Devuelve un diccionario con ambos resultados.
    """
    # ---------------------------------------------------------
    # 1. DISEÑO: Calcular W/H a partir de Z0 (Filas 5, 6, 7 y 8)
    # ---------------------------------------------------------
    # Fila 7: Constante A (Diferente para cada autor)
    A_H = (Z0 / 60.0) * math.sqrt((er + 1.0) / 2.0) + ((er - 1.0) / (er + 1.0)) * (0.23 + 0.11 / er)
    A_W = (Z0 / 60.0) * math.sqrt((er + 1.0) / 2.0) + ((er - 1.0) / (er + 1.0)) * (0.226 + 0.121 / er)
    
    # Fila 8: Constante B (Igual para ambos)
    B = (377.0 * math.pi) / (2.0 * Z0 * math.sqrt(er))
    
    # Fila 5: W/H <= 2 (Cada uno usa su propia constante A)
    WH_1_H = (8.0 * math.exp(A_H)) / (math.exp(2.0 * A_H) - 2.0)
    WH_1_W = (8.0 * math.exp(A_W)) / (math.exp(2.0 * A_W) - 2.0)
    
    # Fila 6: W/H >= 2 (Diferente)
    # -- Hammerstad
    WH_2_H = (2.0 / math.pi) * (B - 1.0 - math.log(2.0 * B - 1.0) + ((er - 1.0) / (2.0 * er)) * (math.log(B - 1.0) + 0.39 - 0.61 / er))
    # -- Wheeler
    termino_1_W = ((er - 1.0) / (math.pi * er)) * (math.log(B - 1.0) + 0.293 - 0.517 / er)
    termino_2_W = (2.0 / math.pi) * (B - 1.0 - math.log(2.0 * B - 1.0))
    WH_2_W = termino_1_W + termino_2_W
    
    # Selección según el límite
    WH_H = WH_1_H if WH_1_H <= 2.0 else WH_2_H
    WH_W = WH_1_W if WH_1_W <= 2.0 else WH_2_W
    
    W_H = WH_H * H
    W_W = WH_W * H

    # ---------------------------------------------------------
    # 2. ANÁLISIS: Calcular Epsilon efectivo (Filas 2 y 4)
    # ---------------------------------------------------------
    def calcular_ere(WH, W):
        if WH <= 1.0:
            return (er + 1.0) / 2.0 + ((er - 1.0) / 2.0) * ((1.0 / math.sqrt(1.0 + 12.0 * H / W)) + 0.04 * (1.0 - WH)**2)
        else:
            return (er + 1.0) / 2.0 + ((er - 1.0) / 2.0) * (1.0 / math.sqrt(1.0 + 12.0 * H / W))

    Ere_H = calcular_ere(WH_H, W_H)
    Ere_W = calcular_ere(WH_W, W_W)

    # ---------------------------------------------------------
    # 3. ANÁLISIS: Comprobación de Z0 (Filas 1 y 3)
    # ---------------------------------------------------------
    # -- Hammerstad
    if WH_H <= 1.0:
        Z0_chk_H = (60.0 / math.sqrt(Ere_H)) * math.log(8.0 / WH_H + WH_H / 4.0)
    else:
        den_H = WH_H + 1.393 + 0.667 * math.log(WH_H + 1.444)
        Z0_chk_H = (120.0 * math.pi / math.sqrt(Ere_H)) / den_H

    # -- Wheeler
    if WH_W <= 1.0:
        Z0_chk_W = (60.0 / math.sqrt(Ere_W)) * math.log(8.0 / WH_W + WH_W / 4.0)
    else:
        den_W = WH_W + 2.46 - 0.49 * (1.0 / WH_W) + (1.0 - 1.0 / WH_W)**6
        Z0_chk_W = (120.0 * math.pi / math.sqrt(Ere_W)) / den_W

    # ---------------------------------------------------------
    # 4. ANÁLISIS: Calcular Ancho Efectivo We (Filas 9 y 10)
    # ---------------------------------------------------------
    limite_wheeler = 1.0 / (2.0 * math.pi)
    def calcular_we(WH, W):
        if WH <= limite_wheeler:
            return W + (t / math.pi) * (1.0 + math.log((4.0 * math.pi * W) / t))
        else:
            return W + (t / math.pi) * (1.0 + math.log((2.0 * H) / t))

    We_H = calcular_we(WH_H, W_H)
    We_W = calcular_we(WH_W, W_W)
        
    return {
        'Hamm': {'W': W_H, 'We': We_H, 'Ere': Ere_H, 'WH': WH_H, 'Z0_chk': Z0_chk_H},
        'Wheel': {'W': W_W, 'We': We_W, 'Ere': Ere_W, 'WH': WH_W, 'Z0_chk': Z0_chk_W}
    }

def calcular_longitud_lambda_cuartos(E_re, f):
    lambda_0 = c / f                             
    lambda_prime = lambda_0 / math.sqrt(E_re)    
    return lambda_prime / 4.0

def calcular_longitud_stub_abierto(C, f, Z0_stub, E_re):
    lambda_0 = c / f                             
    lambda_prime = lambda_0 / math.sqrt(E_re)    
    Beta = 2.0 * math.pi / lambda_prime          
    XC = 1.0 / (2.0 * math.pi * f * C)           
    return math.atan(Z0_stub / XC) / Beta

# ==========================================
# CÁLCULOS Y REPORTE (Función de impresión)
# ==========================================
def imprimir_reporte(titulo, Z_obj, C_obj=None):
    res = calcular_geometria_microtira(Z_obj, er, H, t)
    
    if C_obj is None: # Es un transformador L/4
        L_H = calcular_longitud_lambda_cuartos(res['Hamm']['Ere'], f)
        L_W = calcular_longitud_lambda_cuartos(res['Wheel']['Ere'], f)
    else:             # Es un stub capacitivo
        L_H = calcular_longitud_stub_abierto(C_obj, f, Z_obj, res['Hamm']['Ere'])
        L_W = calcular_longitud_stub_abierto(C_obj, f, Z_obj, res['Wheel']['Ere'])
    
    print(f"{titulo}")
    print(f"Z0 objetivo: {Z_obj} Ohm")
    if C_obj is not None:
        print(f"Capacitancia obj.: {C_obj*1e12:.3f} pF")
    
    print(f"{'PARÁMETRO':<20} | {'HAMMERSTAD':<15} | {'WHEELER':<15}")
    print("-" * 56)
    print(f"{'Relación (W/H)':<20} | {res['Hamm']['WH']:<15.4f} | {res['Wheel']['WH']:<15.4f}")
    print(f"{'Relación (L/W)':<20} | {L_H/res['Hamm']['W']:<15.4f} | {L_W/res['Wheel']['W']:<15.4f}")
    print(f"{'Ancho (W) [mm]':<20} | {res['Hamm']['W']*1000:<15.4f} | {res['Wheel']['W']*1000:<15.4f}")
    print(f"{'Ancho (We) [mm]':<20} | {res['Hamm']['We']*1000:<15.4f} | {res['Wheel']['We']*1000:<15.4f}")
    print(f"{'Epsilon eff (E_re)':<20} | {res['Hamm']['Ere']:<15.4f} | {res['Wheel']['Ere']:<15.4f}")
    print(f"{'Longitud física [mm]':<20} | {L_H*1000:<15.4f} | {L_W*1000:<15.4f}")
    print(f"{'Z0 verificado [Ohm]':<20} | {res['Hamm']['Z0_chk']:<15.4f} | {res['Wheel']['Z0_chk']:<15.4f}\n")

# Ejecución de los reportes
print(f"--- PARÁMETROS DEL SUSTRATO ---")
print(f"Er = {er} | H = {H*1000} mm | t = {t*1e6} um | f = {f/1e9} GHz\n")

imprimir_reporte("--- 1. TRANSFORMADOR L/4 ENTRADA ---", Z_QWin)
imprimir_reporte("--- 2. TRANSFORMADOR L/4 SALIDA ---", Z_QWout)
imprimir_reporte("--- 3. CAPACITOR ENTRADA (STUB ABIERTO) ---", Z0_Comp_in, C_in)
imprimir_reporte("--- 4. CAPACITOR SALIDA (STUB ABIERTO) ---", Z0_Comp_out, C_out)
