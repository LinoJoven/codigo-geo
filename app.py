from flask import Flask, render_template, request, jsonify
import numpy as np

app = Flask(__name__)

# Constantes WGS84
A = 6378137.0
F = 1/298.257223563
B = A * (1 - F)
E2 = 2*F - F**2
EP2 = E2 / (1 - E2)

# Constantes Datum Bogotá
A_BOG = 6378388.0
F_BOG = 1/297.0
B_BOG = A_BOG * (1 - F_BOG)
E2_BOG = 2*F_BOG - F_BOG**2

def dms_to_decimal(g, m, s):
    return g + m/60.0 + s/3600.0

def decimal_to_dms(d):
    g = int(d)
    m_dec = abs((d - g) * 60)
    m = int(m_dec)
    s = (m_dec - m) * 60
    return g, m, s

def validar_latitud(g, m, s):
    if m < 0 or m >= 60:
        raise ValueError("Minutos fuera de rango [0-59]")
    if s < 0 or s >= 60:
        raise ValueError("Segundos fuera de rango [0-59.999]")
    d = dms_to_decimal(g, m, s)
    if d < -90 or d > 90:
        raise ValueError("Latitud fuera de rango [-90°, 90°]")
    return True

def validar_longitud(g, m, s):
    if m < 0 or m >= 60:
        raise ValueError("Minutos fuera de rango [0-59]")
    if s < 0 or s >= 60:
        raise ValueError("Segundos fuera de rango [0-59.999]")
    d = dms_to_decimal(g, m, s)
    if d < -180 or d > 180:
        raise ValueError("Longitud fuera de rango [-180°, 180°]")
    return True

def phi_to_theta(phi):
    return np.arctan((1 - F) * np.tan(phi))

def phi_to_omega(phi):
    return np.arctan((1 - E2) * np.tan(phi))

def theta_to_phi(theta):
    return np.arctan(np.tan(theta) / (1 - F))

def omega_to_phi(omega):
    return np.arctan(np.tan(omega) / (1 - E2))

def phi_lambda_to_xyz(phi_deg, lambda_deg, h=0):
    phi = np.radians(phi_deg)
    lam = np.radians(lambda_deg)
    N = A / np.sqrt(1 - E2 * np.sin(phi)**2)
    X = (N + h) * np.cos(phi) * np.cos(lam)
    Y = (N + h) * np.cos(phi) * np.sin(lam)
    Z = (N * (1 - E2) + h) * np.sin(phi)
    return float(X), float(Y), float(Z)

def xyz_to_phi_lambda_h(X, Y, Z):
    lam = np.arctan2(Y, X)
    p = np.sqrt(X**2 + Y**2)
    phi = np.arctan2(Z, p * (1 - E2))
    for _ in range(10):
        N = A / np.sqrt(1 - E2 * np.sin(phi)**2)
        h = p / np.cos(phi) - N
        phi = np.arctan2(Z, p * (1 - E2 * N / (N + h)))
    return np.degrees(phi), np.degrees(lam), h

def arco_meridiano(phi1, phi2, lam):
    phi1_r, phi2_r = np.radians(phi1), np.radians(phi2)
    lam_r = np.radians(lam)
    
    A_calc = A * (1 - E2)
    e4, e6 = E2**2, E2**3
    c0 = 1 + 3*E2/4 + 45*e4/64 + 175*e6/256
    c2 = 3*E2/4 + 15*e4/16 + 525*e6/512
    c4 = 15*e4/64 + 105*e6/256
    c6 = 35*e6/512
    
    M = lambda phi: A_calc * (c0*phi - c2*np.sin(2*phi)/2 + c4*np.sin(4*phi)/4 - c6*np.sin(6*phi)/6)
    s = M(phi2_r) - M(phi1_r)
    
    puntos = []
    for phi in np.linspace(phi1_r, phi2_r, 100):
        N = A / np.sqrt(1 - E2 * np.sin(phi)**2)
        X = N * np.cos(phi) * np.cos(lam_r)
        Y = N * np.cos(phi) * np.sin(lam_r)
        Z = N * (1 - E2) * np.sin(phi)
        puntos.append([float(X), float(Y), float(Z)])
    
    return abs(s), puntos

def arco_paralelo(phi, lam1, lam2):
    phi_r = np.radians(phi)
    lam1_r, lam2_r = np.radians(lam1), np.radians(lam2)
    N = A / np.sqrt(1 - E2 * np.sin(phi_r)**2)
    r = N * np.cos(phi_r)
    s = r * abs(lam2_r - lam1_r)
    
    puntos = []
    for lam in np.linspace(lam1_r, lam2_r, 100):
        X = N * np.cos(phi_r) * np.cos(lam)
        Y = N * np.cos(phi_r) * np.sin(lam)
        Z = N * (1 - E2) * np.sin(phi_r)
        puntos.append([float(X), float(Y), float(Z)])
    
    return s, puntos

def transform_bogota_wgs84(phi, lam, h):
    dx, dy, dz = 307.0, 304.0, -318.0
    phi_r, lam_r = np.radians(phi), np.radians(lam)
    N = A_BOG / np.sqrt(1 - E2_BOG * np.sin(phi_r)**2)
    
    X = (N + h) * np.cos(phi_r) * np.cos(lam_r) + dx
    Y = (N + h) * np.cos(phi_r) * np.sin(lam_r) + dy
    Z = (N * (1 - E2_BOG) + h) * np.sin(phi_r) + dz
    
    return xyz_to_phi_lambda_h(X, Y, Z)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/apartado/<int:numero>')
def apartado(numero):
    return render_template(f'apartado{numero}.html')

@app.route('/api/apartado1', methods=['POST'])
def api_apartado1():
    try:
        data = request.json
        tipo = data['tipo_entrada']
        g, m, s = float(data['grados']), float(data['minutos']), float(data['segundos'])
        validar_latitud(g, m, s)
        
        val = np.radians(dms_to_decimal(g, m, s))
        
        if tipo == 'phi':
            phi_r, theta_r, omega_r = val, phi_to_theta(val), phi_to_omega(val)
        elif tipo == 'theta':
            theta_r, phi_r, omega_r = val, theta_to_phi(val), phi_to_theta(theta_to_phi(val))
            omega_r = phi_to_omega(phi_r)
        else:  # omega
            omega_r, phi_r = val, omega_to_phi(val)
            theta_r = phi_to_theta(phi_r)
        
        phi_deg, theta_deg, omega_deg = np.degrees(phi_r), np.degrees(theta_r), np.degrees(omega_r)
        X, Y, Z = phi_lambda_to_xyz(phi_deg, 0)
        
        return jsonify({
            'success': True,
            'resultados': {
                'phi': {'decimal': phi_deg, 'dms': decimal_to_dms(phi_deg)},
                'theta': {'decimal': theta_deg, 'dms': decimal_to_dms(theta_deg)},
                'omega': {'decimal': omega_deg, 'dms': decimal_to_dms(omega_deg)},
                'coords': {'X': X, 'Y': Y, 'Z': Z}
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/apartado2', methods=['POST'])
def api_apartado2():
    try:
        data = request.json
        tipo = data['tipo_entrada']
        lat_g, lat_m, lat_s = float(data['lat_grados']), float(data['lat_minutos']), float(data['lat_segundos'])
        lon_g, lon_m, lon_s = float(data['lon_grados']), float(data['lon_minutos']), float(data['lon_segundos'])
        
        validar_latitud(lat_g, lat_m, lat_s)
        validar_longitud(lon_g, lon_m, lon_s)
        
        lat = dms_to_decimal(lat_g, lat_m, lat_s)
        lon = dms_to_decimal(lon_g, lon_m, lon_s)
        
        if tipo == 'theta':
            lat = np.degrees(theta_to_phi(np.radians(lat)))
        elif tipo == 'omega':
            lat = np.degrees(omega_to_phi(np.radians(lat)))
        
        X, Y, Z = phi_lambda_to_xyz(lat, lon)
        return jsonify({'success': True, 'X': X, 'Y': Y, 'Z': Z})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/apartado3', methods=['POST'])
def api_apartado3():
    try:
        data = request.json
        phi_g, phi_m, phi_s = float(data['phi_grados']), float(data['phi_minutos']), float(data['phi_segundos'])
        lam_g, lam_m, lam_s = float(data['lambda_grados']), float(data['lambda_minutos']), float(data['lambda_segundos'])
        h = float(data['h'])
        
        validar_latitud(phi_g, phi_m, phi_s)
        validar_longitud(lam_g, lam_m, lam_s)
        
        phi = dms_to_decimal(phi_g, phi_m, phi_s)
        lam = dms_to_decimal(lam_g, lam_m, lam_s)
        X, Y, Z = phi_lambda_to_xyz(phi, lam, h)
        
        return jsonify({'success': True, 'X': X, 'Y': Y, 'Z': Z})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/apartado4', methods=['POST'])
def api_apartado4():
    try:
        X, Y, Z = float(request.json['X']), float(request.json['Y']), float(request.json['Z'])
        phi, lam, h = xyz_to_phi_lambda_h(X, Y, Z)
        
        return jsonify({
            'success': True,
            'phi': {'decimal': phi, 'dms': decimal_to_dms(phi)},
            'lambda': {'decimal': lam, 'dms': decimal_to_dms(lam)},
            'h': h,
            'coords': {'X': X, 'Y': Y, 'Z': Z}
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/apartado5', methods=['POST'])
def api_apartado5():
    try:
        data = request.json
        phi1_g, phi1_m, phi1_s = float(data['phi1_grados']), float(data['phi1_minutos']), float(data['phi1_segundos'])
        phi2_g, phi2_m, phi2_s = float(data['phi2_grados']), float(data['phi2_minutos']), float(data['phi2_segundos'])
        lam_g, lam_m, lam_s = float(data['lambda_grados']), float(data['lambda_minutos']), float(data['lambda_segundos'])
        
        validar_latitud(phi1_g, phi1_m, phi1_s)
        validar_latitud(phi2_g, phi2_m, phi2_s)
        validar_longitud(lam_g, lam_m, lam_s)
        
        phi1 = dms_to_decimal(phi1_g, phi1_m, phi1_s)
        phi2 = dms_to_decimal(phi2_g, phi2_m, phi2_s)
        lam = dms_to_decimal(lam_g, lam_m, lam_s)
        
        long, puntos = arco_meridiano(phi1, phi2, lam)
        return jsonify({'success': True, 'longitud': long, 'puntos': puntos})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/apartado6', methods=['POST'])
def api_apartado6():
    try:
        data = request.json
        phi_g, phi_m, phi_s = float(data['phi_grados']), float(data['phi_minutos']), float(data['phi_segundos'])
        lam1_g, lam1_m, lam1_s = float(data['lambda1_grados']), float(data['lambda1_minutos']), float(data['lambda1_segundos'])
        lam2_g, lam2_m, lam2_s = float(data['lambda2_grados']), float(data['lambda2_minutos']), float(data['lambda2_segundos'])
        
        validar_latitud(phi_g, phi_m, phi_s)
        validar_longitud(lam1_g, lam1_m, lam1_s)
        validar_longitud(lam2_g, lam2_m, lam2_s)
        
        phi = dms_to_decimal(phi_g, phi_m, phi_s)
        lam1 = dms_to_decimal(lam1_g, lam1_m, lam1_s)
        lam2 = dms_to_decimal(lam2_g, lam2_m, lam2_s)
        
        long, puntos = arco_paralelo(phi, lam1, lam2)
        return jsonify({'success': True, 'longitud': long, 'puntos': puntos})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/apartado7', methods=['POST'])
def api_apartado7():
    try:
        data = request.json
        cota = data['cota_inicial']
        resultados = []
        
        for i, p in enumerate(data['puntos']):
            desnivel = p.get('v_mas', 0) - p.get('v_menos', 0)
            cota += desnivel
            resultados.append({
                'estacion': i + 1,
                'v_mas': p.get('v_mas', 0),
                'v_menos': p.get('v_menos', 0),
                'v_intermedia': p.get('v_intermedia', None),
                'desnivel': desnivel,
                'cota': cota,
                'distancia': p.get('distancia', 0)
            })
        
        return jsonify({'success': True, 'resultados': resultados})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/apartado8', methods=['POST'])
def api_apartado8():
    try:
        data = request.json
        phi_g, phi_m, phi_s = float(data['phi_grados']), float(data['phi_minutos']), float(data['phi_segundos'])
        lam_g, lam_m, lam_s = float(data['lambda_grados']), float(data['lambda_minutos']), float(data['lambda_segundos'])
        h = float(data['h'])
        
        validar_latitud(phi_g, phi_m, phi_s)
        validar_longitud(lam_g, lam_m, lam_s)
        
        phi = dms_to_decimal(phi_g, phi_m, phi_s)
        lam = dms_to_decimal(lam_g, lam_m, lam_s)
        
        phi_wgs, lam_wgs, h_wgs = transform_bogota_wgs84(phi, lam, h)
        
        return jsonify({
            'success': True,
            'wgs84': {
                'phi': {'decimal': phi_wgs, 'dms': decimal_to_dms(phi_wgs)},
                'lambda': {'decimal': lam_wgs, 'dms': decimal_to_dms(lam_wgs)},
                'h': h_wgs
            },
            'grs80': {
                'phi': {'decimal': phi_wgs, 'dms': decimal_to_dms(phi_wgs)},
                'lambda': {'decimal': lam_wgs, 'dms': decimal_to_dms(lam_wgs)},
                'h': h_wgs
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)