from flask import Flask, render_template, request, jsonify
import numpy as np
import math

app = Flask(__name__)

# Parámetros del Elipsoide WGS84
a_wgs84 = 6378137.0  # Semi-eje mayor
f_wgs84 = 1/298.257223563  # Achatamiento
b_wgs84 = a_wgs84 * (1 - f_wgs84)  # Semi-eje menor
e2_wgs84 = 2*f_wgs84 - f_wgs84**2  # Primera excentricidad al cuadrado
ep2_wgs84 = e2_wgs84 / (1 - e2_wgs84)  # Segunda excentricidad al cuadrado

# Parámetros del Elipsoide GRS80
a_grs80 = 6378137.0
f_grs80 = 1/298.257222101
b_grs80 = a_grs80 * (1 - f_grs80)
e2_grs80 = 2*f_grs80 - f_grs80**2
ep2_grs80 = e2_grs80 / (1 - e2_grs80)

# Parámetros del Datum Bogotá (Internacional 1924)
a_bogota = 6378388.0
f_bogota = 1/297.0
b_bogota = a_bogota * (1 - f_bogota)
e2_bogota = 2*f_bogota - f_bogota**2

# Funciones auxiliares de conversión
def dms_to_decimal(grados, minutos, segundos):
    """Convierte grados, minutos, segundos a grados decimales"""
    return grados + minutos/60.0 + segundos/3600.0

def decimal_to_dms(decimal):
    """Convierte grados decimales a grados, minutos, segundos"""
    grados = int(decimal)
    minutos_decimal = abs((decimal - grados) * 60)
    minutos = int(minutos_decimal)
    segundos = (minutos_decimal - minutos) * 60
    return grados, minutos, segundos

# APARTADO 1: Transformación de Latitudes
def geodesica_a_parametrica(phi_rad):
    """Convierte latitud geodésica a paramétrica"""
    theta_rad = np.arctan((1 - f_wgs84) * np.tan(phi_rad))
    return theta_rad

def geodesica_a_geocentrica(phi_rad):
    """Convierte latitud geodésica a geocéntrica"""
    omega_rad = np.arctan((1 - e2_wgs84) * np.tan(phi_rad))
    return omega_rad

def parametrica_a_geodesica(theta_rad):
    """Convierte latitud paramétrica a geodésica"""
    phi_rad = np.arctan(np.tan(theta_rad) / (1 - f_wgs84))
    return phi_rad

def parametrica_a_geocentrica(theta_rad):
    """Convierte latitud paramétrica a geocéntrica"""
    phi_rad = parametrica_a_geodesica(theta_rad)
    omega_rad = geodesica_a_geocentrica(phi_rad)
    return omega_rad

def geocentrica_a_geodesica(omega_rad):
    """Convierte latitud geocéntrica a geodésica"""
    phi_rad = np.arctan(np.tan(omega_rad) / (1 - e2_wgs84))
    return phi_rad

def geocentrica_a_parametrica(omega_rad):
    """Convierte latitud geocéntrica a paramétrica"""
    phi_rad = geocentrica_a_geodesica(omega_rad)
    theta_rad = geodesica_a_parametrica(phi_rad)
    return theta_rad

# APARTADO 2: Coordenadas con Lambda
def theta_lambda_to_xyz(theta_deg, lambda_deg):
    """Convierte (θ, λ) a (X, Y, Z)"""
    theta_rad = np.radians(theta_deg)
    lambda_rad = np.radians(lambda_deg)
    
    phi_rad = parametrica_a_geodesica(theta_rad)
    N = a_wgs84 / np.sqrt(1 - e2_wgs84 * np.sin(phi_rad)**2)
    
    X = N * np.cos(phi_rad) * np.cos(lambda_rad)
    Y = N * np.cos(phi_rad) * np.sin(lambda_rad)
    Z = N * (1 - e2_wgs84) * np.sin(phi_rad)
    
    return X, Y, Z

def phi_lambda_to_xyz(phi_deg, lambda_deg):
    """Convierte (φ, λ) a (X, Y, Z)"""
    phi_rad = np.radians(phi_deg)
    lambda_rad = np.radians(lambda_deg)
    
    N = a_wgs84 / np.sqrt(1 - e2_wgs84 * np.sin(phi_rad)**2)
    
    X = N * np.cos(phi_rad) * np.cos(lambda_rad)
    Y = N * np.cos(phi_rad) * np.sin(lambda_rad)
    Z = N * (1 - e2_wgs84) * np.sin(phi_rad)
    
    return X, Y, Z

def omega_lambda_to_xyz(omega_deg, lambda_deg):
    """Convierte (ω, λ) a (X, Y, Z)"""
    omega_rad = np.radians(omega_deg)
    lambda_rad = np.radians(lambda_deg)
    
    phi_rad = geocentrica_a_geodesica(omega_rad)
    N = a_wgs84 / np.sqrt(1 - e2_wgs84 * np.sin(phi_rad)**2)
    
    X = N * np.cos(phi_rad) * np.cos(lambda_rad)
    Y = N * np.cos(phi_rad) * np.sin(lambda_rad)
    Z = N * (1 - e2_wgs84) * np.sin(phi_rad)
    
    return X, Y, Z

# APARTADO 3: Coordenadas con altura
def phi_lambda_h_to_xyz(phi_deg, lambda_deg, h):
    """Convierte (φ, λ, h) a (X, Y, Z)"""
    phi_rad = np.radians(phi_deg)
    lambda_rad = np.radians(lambda_deg)
    
    N = a_wgs84 / np.sqrt(1 - e2_wgs84 * np.sin(phi_rad)**2)
    
    X = (N + h) * np.cos(phi_rad) * np.cos(lambda_rad)
    Y = (N + h) * np.cos(phi_rad) * np.sin(lambda_rad)
    Z = (N * (1 - e2_wgs84) + h) * np.sin(phi_rad)
    
    return X, Y, Z

# APARTADO 4: Problema inverso
def xyz_to_phi_lambda_h(X, Y, Z):
    """Convierte (X, Y, Z) a (φ, λ, h)"""
    lambda_rad = np.arctan2(Y, X)
    p = np.sqrt(X**2 + Y**2)
    
    # Iteración para φ
    phi_rad = np.arctan2(Z, p * (1 - e2_wgs84))
    for _ in range(10):
        N = a_wgs84 / np.sqrt(1 - e2_wgs84 * np.sin(phi_rad)**2)
        h = p / np.cos(phi_rad) - N
        phi_rad = np.arctan2(Z, p * (1 - e2_wgs84 * N / (N + h)))
    
    phi_deg = np.degrees(phi_rad)
    lambda_deg = np.degrees(lambda_rad)
    
    return phi_deg, lambda_deg, h

# APARTADO 5: Longitud de arco de meridiano
def longitud_arco_meridiano(phi1_deg, phi2_deg, lambda_deg):
    """Calcula la longitud del arco de meridiano entre phi1 y phi2"""
    phi1_rad = np.radians(phi1_deg)
    phi2_rad = np.radians(phi2_deg)
    
    # Fórmula de la longitud del arco de meridiano
    A = a_wgs84 * (1 - e2_wgs84)
    e4 = e2_wgs84 ** 2
    e6 = e2_wgs84 ** 3
    
    # Coeficientes
    c0 = 1 + 3*e2_wgs84/4 + 45*e4/64 + 175*e6/256
    c2 = 3*e2_wgs84/4 + 15*e4/16 + 525*e6/512
    c4 = 15*e4/64 + 105*e6/256
    c6 = 35*e6/512
    
    def M(phi):
        return A * (c0*phi - c2*np.sin(2*phi)/2 + c4*np.sin(4*phi)/4 - c6*np.sin(6*phi)/6)
    
    s = M(phi2_rad) - M(phi1_rad)
    
    # Puntos para la gráfica
    num_points = 100
    phis = np.linspace(phi1_rad, phi2_rad, num_points)
    lambda_rad = np.radians(lambda_deg)
    
    points = []
    for phi in phis:
        N = a_wgs84 / np.sqrt(1 - e2_wgs84 * np.sin(phi)**2)
        X = N * np.cos(phi) * np.cos(lambda_rad)
        Y = N * np.cos(phi) * np.sin(lambda_rad)
        Z = N * (1 - e2_wgs84) * np.sin(phi)
        points.append([X, Y, Z])
    
    return abs(s), points

# APARTADO 6: Longitud de arco de paralelo
def longitud_arco_paralelo(phi_deg, lambda1_deg, lambda2_deg):
    """Calcula la longitud del arco de paralelo"""
    phi_rad = np.radians(phi_deg)
    lambda1_rad = np.radians(lambda1_deg)
    lambda2_rad = np.radians(lambda2_deg)
    
    N = a_wgs84 / np.sqrt(1 - e2_wgs84 * np.sin(phi_rad)**2)
    r = N * np.cos(phi_rad)
    
    delta_lambda = abs(lambda2_rad - lambda1_rad)
    s = r * delta_lambda
    
    # Puntos para la gráfica
    num_points = 100
    lambdas = np.linspace(lambda1_rad, lambda2_rad, num_points)
    
    points = []
    for lam in lambdas:
        X = N * np.cos(phi_rad) * np.cos(lam)
        Y = N * np.cos(phi_rad) * np.sin(lam)
        Z = N * (1 - e2_wgs84) * np.sin(phi_rad)
        points.append([X, Y, Z])
    
    return s, points

# APARTADO 7: Nivelación diferencial
def calcular_nivelacion(datos):
    """Calcula la nivelación diferencial"""
    resultados = []
    cota_actual = datos['cota_inicial']
    
    for i, punto in enumerate(datos['puntos']):
        v_mas = punto.get('v_mas', 0)
        v_menos = punto.get('v_menos', 0)
        
        desnivel = v_mas - v_menos
        cota_actual += desnivel
        
        resultados.append({
            'estacion': i + 1,
            'v_mas': v_mas,
            'v_menos': v_menos,
            'v_intermedia': punto.get('v_intermedia', None),
            'desnivel': desnivel,
            'cota': cota_actual,
            'distancia': punto.get('distancia', 0)
        })
    
    return resultados

# APARTADO 8: Transformación de Datums
def transformar_datum_bogota_a_wgs84(phi_deg, lambda_deg, h):
    """Transforma de Datum Bogotá a WGS84"""
    # Parámetros de transformación (valores aproximados para Colombia)
    dx = 307.0  # metros
    dy = 304.0
    dz = -318.0
    
    # Convertir a cartesianas en Bogotá
    phi_rad = np.radians(phi_deg)
    lambda_rad = np.radians(lambda_deg)
    
    N_bog = a_bogota / np.sqrt(1 - e2_bogota * np.sin(phi_rad)**2)
    
    X_bog = (N_bog + h) * np.cos(phi_rad) * np.cos(lambda_rad)
    Y_bog = (N_bog + h) * np.cos(phi_rad) * np.sin(lambda_rad)
    Z_bog = (N_bog * (1 - e2_bogota) + h) * np.sin(phi_rad)
    
    # Aplicar traslación
    X_wgs = X_bog + dx
    Y_wgs = Y_bog + dy
    Z_wgs = Z_bog + dz
    
    # Convertir a geodésicas WGS84
    phi_wgs, lambda_wgs, h_wgs = xyz_to_phi_lambda_h(X_wgs, Y_wgs, Z_wgs)
    
    return phi_wgs, lambda_wgs, h_wgs

def transformar_datum_bogota_a_grs80(phi_deg, lambda_deg, h):
    """Transforma de Datum Bogotá a GRS80"""
    # Similar a WGS84 con parámetros GRS80
    phi_wgs, lambda_wgs, h_wgs = transformar_datum_bogota_a_wgs84(phi_deg, lambda_deg, h)
    return phi_wgs, lambda_wgs, h_wgs

# RUTAS
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/apartado/<int:numero>')
def apartado(numero):
    return render_template(f'apartado{numero}.html')

# APIs para cada apartado
@app.route('/api/apartado1', methods=['POST'])
def api_apartado1():
    try:
        data = request.json
        tipo_entrada = data['tipo_entrada']
        grados = float(data['grados'])
        minutos = float(data['minutos'])
        segundos = float(data['segundos'])
        
        valor_decimal = dms_to_decimal(grados, minutos, segundos)
        valor_rad = np.radians(valor_decimal)
        
        resultados = {}
        
        if tipo_entrada == 'phi':
            theta_rad = geodesica_a_parametrica(valor_rad)
            omega_rad = geodesica_a_geocentrica(valor_rad)
            
            theta_deg = np.degrees(theta_rad)
            omega_deg = np.degrees(omega_rad)
            
            resultados['phi'] = {'decimal': valor_decimal, 'dms': decimal_to_dms(valor_decimal)}
            resultados['theta'] = {'decimal': theta_deg, 'dms': decimal_to_dms(theta_deg)}
            resultados['omega'] = {'decimal': omega_deg, 'dms': decimal_to_dms(omega_deg)}
            
        elif tipo_entrada == 'theta':
            phi_rad = parametrica_a_geodesica(valor_rad)
            omega_rad = parametrica_a_geocentrica(valor_rad)
            
            phi_deg = np.degrees(phi_rad)
            omega_deg = np.degrees(omega_rad)
            
            resultados['phi'] = {'decimal': phi_deg, 'dms': decimal_to_dms(phi_deg)}
            resultados['theta'] = {'decimal': valor_decimal, 'dms': decimal_to_dms(valor_decimal)}
            resultados['omega'] = {'decimal': omega_deg, 'dms': decimal_to_dms(omega_deg)}
            
        elif tipo_entrada == 'omega':
            phi_rad = geocentrica_a_geodesica(valor_rad)
            theta_rad = geocentrica_a_parametrica(valor_rad)
            
            phi_deg = np.degrees(phi_rad)
            theta_deg = np.degrees(theta_rad)
            
            resultados['phi'] = {'decimal': phi_deg, 'dms': decimal_to_dms(phi_deg)}
            resultados['theta'] = {'decimal': theta_deg, 'dms': decimal_to_dms(theta_deg)}
            resultados['omega'] = {'decimal': valor_decimal, 'dms': decimal_to_dms(valor_decimal)}
        
        return jsonify({'success': True, 'resultados': resultados})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/apartado2', methods=['POST'])
def api_apartado2():
    try:
        data = request.json
        tipo_entrada = data['tipo_entrada']
        
        lat_g = float(data['lat_grados'])
        lat_m = float(data['lat_minutos'])
        lat_s = float(data['lat_segundos'])
        lat_decimal = dms_to_decimal(lat_g, lat_m, lat_s)
        
        lon_g = float(data['lon_grados'])
        lon_m = float(data['lon_minutos'])
        lon_s = float(data['lon_segundos'])
        lon_decimal = dms_to_decimal(lon_g, lon_m, lon_s)
        
        if tipo_entrada == 'theta':
            X, Y, Z = theta_lambda_to_xyz(lat_decimal, lon_decimal)
        elif tipo_entrada == 'phi':
            X, Y, Z = phi_lambda_to_xyz(lat_decimal, lon_decimal)
        elif tipo_entrada == 'omega':
            X, Y, Z = omega_lambda_to_xyz(lat_decimal, lon_decimal)
        
        return jsonify({'success': True, 'X': X, 'Y': Y, 'Z': Z})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/apartado3', methods=['POST'])
def api_apartado3():
    try:
        data = request.json
        
        phi_g = float(data['phi_grados'])
        phi_m = float(data['phi_minutos'])
        phi_s = float(data['phi_segundos'])
        phi_decimal = dms_to_decimal(phi_g, phi_m, phi_s)
        
        lambda_g = float(data['lambda_grados'])
        lambda_m = float(data['lambda_minutos'])
        lambda_s = float(data['lambda_segundos'])
        lambda_decimal = dms_to_decimal(lambda_g, lambda_m, lambda_s)
        
        h = float(data['h'])
        
        X, Y, Z = phi_lambda_h_to_xyz(phi_decimal, lambda_decimal, h)
        
        return jsonify({'success': True, 'X': X, 'Y': Y, 'Z': Z})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/apartado4', methods=['POST'])
def api_apartado4():
    try:
        data = request.json
        X = float(data['X'])
        Y = float(data['Y'])
        Z = float(data['Z'])
        
        phi_deg, lambda_deg, h = xyz_to_phi_lambda_h(X, Y, Z)
        
        phi_dms = decimal_to_dms(phi_deg)
        lambda_dms = decimal_to_dms(lambda_deg)
        
        return jsonify({
            'success': True,
            'phi': {'decimal': phi_deg, 'dms': phi_dms},
            'lambda': {'decimal': lambda_deg, 'dms': lambda_dms},
            'h': h
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/apartado5', methods=['POST'])
def api_apartado5():
    try:
        data = request.json
        
        phi1_g = float(data['phi1_grados'])
        phi1_m = float(data['phi1_minutos'])
        phi1_s = float(data['phi1_segundos'])
        phi1_decimal = dms_to_decimal(phi1_g, phi1_m, phi1_s)
        
        phi2_g = float(data['phi2_grados'])
        phi2_m = float(data['phi2_minutos'])
        phi2_s = float(data['phi2_segundos'])
        phi2_decimal = dms_to_decimal(phi2_g, phi2_m, phi2_s)
        
        lambda_g = float(data['lambda_grados'])
        lambda_m = float(data['lambda_minutos'])
        lambda_s = float(data['lambda_segundos'])
        lambda_decimal = dms_to_decimal(lambda_g, lambda_m, lambda_s)
        
        longitud, puntos = longitud_arco_meridiano(phi1_decimal, phi2_decimal, lambda_decimal)
        
        return jsonify({
            'success': True,
            'longitud': longitud,
            'puntos': puntos
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/apartado6', methods=['POST'])
def api_apartado6():
    try:
        data = request.json
        
        phi_g = float(data['phi_grados'])
        phi_m = float(data['phi_minutos'])
        phi_s = float(data['phi_segundos'])
        phi_decimal = dms_to_decimal(phi_g, phi_m, phi_s)
        
        lambda1_g = float(data['lambda1_grados'])
        lambda1_m = float(data['lambda1_minutos'])
        lambda1_s = float(data['lambda1_segundos'])
        lambda1_decimal = dms_to_decimal(lambda1_g, lambda1_m, lambda1_s)
        
        lambda2_g = float(data['lambda2_grados'])
        lambda2_m = float(data['lambda2_minutos'])
        lambda2_s = float(data['lambda2_segundos'])
        lambda2_decimal = dms_to_decimal(lambda2_g, lambda2_m, lambda2_s)
        
        longitud, puntos = longitud_arco_paralelo(phi_decimal, lambda1_decimal, lambda2_decimal)
        
        return jsonify({
            'success': True,
            'longitud': longitud,
            'puntos': puntos
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/apartado7', methods=['POST'])
def api_apartado7():
    try:
        data = request.json
        resultados = calcular_nivelacion(data)
        
        return jsonify({'success': True, 'resultados': resultados})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/apartado8', methods=['POST'])
def api_apartado8():
    try:
        data = request.json
        
        phi_g = float(data['phi_grados'])
        phi_m = float(data['phi_minutos'])
        phi_s = float(data['phi_segundos'])
        phi_decimal = dms_to_decimal(phi_g, phi_m, phi_s)
        
        lambda_g = float(data['lambda_grados'])
        lambda_m = float(data['lambda_minutos'])
        lambda_s = float(data['lambda_segundos'])
        lambda_decimal = dms_to_decimal(lambda_g, lambda_m, lambda_s)
        
        h = float(data['h'])
        
        phi_wgs84, lambda_wgs84, h_wgs84 = transformar_datum_bogota_a_wgs84(phi_decimal, lambda_decimal, h)
        phi_grs80, lambda_grs80, h_grs80 = transformar_datum_bogota_a_grs80(phi_decimal, lambda_decimal, h)
        
        return jsonify({
            'success': True,
            'wgs84': {
                'phi': {'decimal': phi_wgs84, 'dms': decimal_to_dms(phi_wgs84)},
                'lambda': {'decimal': lambda_wgs84, 'dms': decimal_to_dms(lambda_wgs84)},
                'h': h_wgs84
            },
            'grs80': {
                'phi': {'decimal': phi_grs80, 'dms': decimal_to_dms(phi_grs80)},
                'lambda': {'decimal': lambda_grs80, 'dms': decimal_to_dms(lambda_grs80)},
                'h': h_grs80
            }
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)