import skrf as rf
import pandas as pd
import numpy as np
import os
import warnings
import re

# Configuramos pandas para la terminal
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.width', 3000)

warnings.filterwarnings("ignore", category=RuntimeWarning)

carpeta_modelos = r'C:\Users\pablo\Desktop\data' 

def format_complex(z):
    r, x = np.real(z), np.imag(z)
    return f" {r:.2f}{'+j' if x>=0 else '-j'}{abs(x):.2f}"

if not os.path.exists(carpeta_modelos):
    print(f"¡CUIDADO! Python no encuentra la carpeta: {carpeta_modelos}")
else:
    transistores = rf.io.read_all(carpeta_modelos)
    datos_polarizacion = []
    
    f_objetivo = 2e9      # 2 GHz
    w0 = 2 * np.pi * f_objetivo
    Z0 = 50.0             # Impedancia del sistema
    R_gen = 50.0          # Resistencia del generador
    R_L = 50.0            # Resistencia de la carga
    
    for nombre_archivo, transistor in transistores.items():
        if "BFP520" not in nombre_archivo.upper():
            continue
            
        try:
            indice_f = np.argmin(np.abs(transistor.f - f_objetivo))
            
            s = transistor.s[indice_f]
            s11, s12 = s[0,0], s[0,1]
            s21, s22 = s[1,0], s[1,1]
            
            mag_S21 = np.abs(s21)
            mag_S12 = np.abs(s12)
            
            delta = s11*s22 - s12*s21
            mag_delta = np.abs(delta)
            
            k_factor = float(np.real(transistor.stability[indice_f]))
            
            if k_factor > 1 and mag_delta < 1:
                termino_k = k_factor - np.sqrt(k_factor**2 - 1)
                gt_max_lineal = (mag_S21 / mag_S12) * termino_k
                ganancia_db = 10 * np.log10(gt_max_lineal)
                tipo_g = "GT(max)"
                
                B1 = 1 + np.abs(s11)**2 - np.abs(s22)**2 - mag_delta**2
                B2 = 1 + np.abs(s22)**2 - np.abs(s11)**2 - mag_delta**2
                C1 = s11 - delta * np.conj(s22)
                C2 = s22 - delta * np.conj(s11)
                
                signo_B1 = 1 if B1 > 0 else -1
                signo_B2 = 1 if B2 > 0 else -1
                
                rho_s = (B1 - signo_B1 * np.sqrt(B1**2 - 4 * np.abs(C1)**2)) / (2 * C1)
                rho_l = (B2 - signo_B2 * np.sqrt(B2**2 - 4 * np.abs(C2)**2)) / (2 * C2)
                
                mag_rho_s, ang_rho_s = np.round(np.abs(rho_s), 3), np.round(np.angle(rho_s, deg=True), 1)
                mag_rho_l, ang_rho_l = np.round(np.abs(rho_l), 3), np.round(np.angle(rho_l, deg=True), 1)
                
                rho_in = np.conj(rho_s)
                rho_out = np.conj(rho_l)
                
                Z_in = Z0 * (1 + rho_in) / (1 - rho_in)
                Z_out = Z0 * (1 + rho_out) / (1 - rho_out)
                
                str_zin = format_complex(Z_in)
                str_zout = format_complex(Z_out)
                
                # ==========================================
                # --- RED DE ENTRADA (LÓGICA DINÁMICA) ---
                # ==========================================
                R_in_s = np.real(Z_in)
                X_in_s = np.imag(Z_in)
                
                if X_in_s > 0:
                    # Reactancia Inductiva -> Cancelamos ANTES del QW
                    top_in = "Antes L/4"
                    R_in_p = R_in_s * (1 + (X_in_s / R_in_s)**2) if R_in_s != 0 else np.inf
                    X_to_cancel_in = X_in_s * (1 + (R_in_s / X_in_s)**2) if X_in_s != 0 else np.inf
                    Z0_QWin = np.sqrt(R_gen * R_in_p) if R_in_p >= 0 else np.nan
                else:
                    # Reactancia Capacitiva -> Cancelamos DESPUÉS del QW
                    top_in = "Despues L/4"
                    Z0_QWin = np.sqrt(R_gen * R_in_s) if R_in_s >= 0 else np.nan
                    # La linea L/4 invierte la admitancia
                    X_to_cancel_in = -(Z0_QWin**2) / X_in_s if X_in_s != 0 else np.inf
                
                str_z0_qwin = f"{Z0_QWin:.2f}"
                str_xin_p = f" j{X_to_cancel_in:.2f}" if X_to_cancel_in >= 0 else f" -j{abs(X_to_cancel_in):.2f}"
                
                # Reactancia de adaptación necesaria
                X_adap_in = -X_to_cancel_in
                
                if X_adap_in < 0:
                    C_val_in = 1 / (w0 * abs(X_adap_in))
                    str_comp_in = f"{C_val_in * 1e12:.3f} pF"
                elif X_adap_in > 0:
                    L_val_in = abs(X_adap_in) / w0
                    str_comp_in = f"{L_val_in * 1e9:.3f} nH"
                else:
                    str_comp_in = "0"

                # ==========================================
                # --- RED DE SALIDA (LÓGICA DINÁMICA) ---
                # ==========================================
                R_out_s = np.real(Z_out)
                X_out_s = np.imag(Z_out)

                if X_out_s < 0:
                    # Reactancia Capacitiva -> Cancelamos DESPUÉS del QW
                    top_out = "Despues L/4"
                    Z0_QWout = np.sqrt(R_L * R_out_s) if R_out_s >= 0 else np.nan
                    # La linea L/4 invierte la admitancia
                    X_to_cancel_out = -(Z0_QWout**2) / X_out_s if X_out_s != 0 else np.inf
                else:
                    # Reactancia Inductiva -> Cancelamos ANTES del QW
                    top_out = "Antes L/4"
                    R_out_p = R_out_s * (1 + (X_out_s / R_out_s)**2) if R_out_s != 0 else np.inf
                    X_to_cancel_out = X_out_s * (1 + (R_out_s / X_out_s)**2) if X_out_s != 0 else np.inf
                    Z0_QWout = np.sqrt(R_L * R_out_p) if R_out_p >= 0 else np.nan

                str_z0_qwout = f"{Z0_QWout:.2f}"
                str_xout_p = f" j{X_to_cancel_out:.2f}" if X_to_cancel_out >= 0 else f" -j{abs(X_to_cancel_out):.2f}"

                # Reactancia de adaptación necesaria
                X_adap_out = -X_to_cancel_out
                
                if X_adap_out < 0:
                    C_val_out = 1 / (w0 * abs(X_adap_out))
                    str_comp_out = f"{C_val_out * 1e12:.3f} pF"
                elif X_adap_out > 0:
                    L_val_out = abs(X_adap_out) / w0
                    str_comp_out = f"{L_val_out * 1e9:.3f} nH"
                else:
                    str_comp_out = "0"
                
            else:
                mgs_lineal = mag_S21 / mag_S12
                ganancia_db = 10 * np.log10(mgs_lineal)
                tipo_g = "MGS"
                mag_rho_s = ang_rho_s = mag_rho_l = ang_rho_l = "N/A"
                str_zin = str_zout = "N/A"
                top_in = str_xin_p = str_comp_in = str_z0_qwin = "N/A"
                top_out = str_xout_p = str_comp_out = str_z0_qwout = "N/A"
            
            vce_val = ic_val = 0.0
            match_vce = re.search(r'VCE_([0-9\.]+)V', nombre_archivo.upper())
            if match_vce: vce_val = float(match_vce.group(1))
            match_ic = re.search(r'IC_([0-9\.]+)(MA|A)', nombre_archivo.upper())
            if match_ic:
                ic_val = float(match_ic.group(1))
                if match_ic.group(2) == 'A': ic_val *= 1000.0  
            
            datos_polarizacion.append({
                'Archivo': nombre_archivo,
                'VCE_Num': vce_val, 'IC_Num': ic_val,
                '|S11|': np.round(np.abs(s11), 3), 'aS11': np.round(np.angle(s11, deg=True), 0),
                '|S21|': np.round(np.abs(s21), 3), 'aS21': np.round(np.angle(s21, deg=True), 0),
                '|S12|': np.round(np.abs(s12), 3), 'aS12': np.round(np.angle(s12, deg=True), 0),
                '|S22|': np.round(np.abs(s22), 3), 'aS22': np.round(np.angle(s22, deg=True), 0),
                'K': np.round(k_factor, 3), '|D|': np.round(mag_delta, 3),
                'G(dB)': np.round(ganancia_db, 2), 'Tipo': tipo_g,
                '|rS|': mag_rho_s, 'arS': ang_rho_s,
                '|rL|': mag_rho_l, 'arL': ang_rho_l,
                'Zin(s)': str_zin, 'Zout(s)': str_zout,
                'Top_in': top_in, 'Xin(p)': str_xin_p, 'Comp_in': str_comp_in, 'Z0_QWin(Ohm)': str_z0_qwin,
                'Top_out': top_out, 'Xout(p)': str_xout_p, 'Comp_out': str_comp_out, 'Z0_QWout(Ohm)': str_z0_qwout 
            })
        except Exception as e:
            print(f"Error en {nombre_archivo}: {e}")
            
    if datos_polarizacion:
        df = pd.DataFrame(datos_polarizacion)
        df = df.sort_values(by=['VCE_Num', 'IC_Num'], ascending=[True, True])
        
        # Actualizamos las columnas a mostrar
        cols = ['Archivo', '|S11|', 'aS11', '|S21|', 'aS21', '|S12|', 'aS12', '|S22|', 'aS22', 
                'K', '|D|', 'G(dB)', 'Tipo', '|rS|', 'arS', '|rL|', 'arL', 
                'Zin(s)', 'Zout(s)', 
                'Top_in', 'Xin(p)', 'Comp_in', 'Z0_QWin(Ohm)',
                'Top_out', 'Xout(p)', 'Comp_out', 'Z0_QWout(Ohm)']
        
        print(f"\n--- REPORTE FINAL DE DISEÑO ({f_objetivo/1e9} GHz) ---")
        print(df[cols].to_string(index=False))
        
        df_estables = df[(df['K'] > 1) & (df['|D|'] < 1)].copy()
        df_estables = df_estables.sort_values(by='G(dB)', ascending=False)
        
        try:
            df[cols].to_csv('BFP520_Tabla_Completa.csv', index=False, sep=';', decimal=',')
            df_estables[cols].to_csv('BFP520_Solo_Estables.csv', index=False, sep=';', decimal=',')
            print("\n✅ ¡SISTEMA COMPLETADO! Se actualizaron los CSV.")
            print("El algoritmo ahora selecciona la topología óptima (cancelar antes o después del L/4) según el signo de la reactancia para cada puerto.")
        except Exception as e:
            print(f"\nNo se pudieron guardar los CSV. Detalle: {e}")
