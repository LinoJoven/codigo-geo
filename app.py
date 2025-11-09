from flask import Flask, render_template, request, jsonify
import numpy as np

app = Flask(__name__)

# ============================================
# CONSTANTES WGS84
# ============================================
A = 6378137.0
F = 1/298.257223563
B = A * (1 - F)
E2 = 2*F - F**2
EP2 = E2 / (1 - E2)

# ============================================
# FUNCIONES AUXILIARES
# ============================================
def dms_to_decimal(g, m, s):
    """Convierte grados, minutos, segundos a decimal"""
    return g + m/60.0 + s/3600.0

def decimal_to_dms(d):
    """Convierte decimal a DMS"""
    g = int(d)
    m_dec = abs((d - g) * 60)
    m = int(m_dec)
    s = (m_dec - m) * 60
    return g, m, s

def validar_latitud(g, m, s):
    """Valida rangos geodésicos de latitud"""
    if m < 0 or m >= 60:
        raise ValueError("Minutos fuera de rango [0-59]")
    if s < 0 or s >= 60:
        raise ValueError("Segundos fuera de rango [0-59.999]")
    d = dms_to_decimal(g, m, s)
    if d < -90 or d > 90:
        raise ValueError("Latitud fuera de rango [-90°, 90°]")
    return True

def validar_longitud(g, m, s):
    """Valida rangos geodésicos de longitud"""
    if m < 0 or m >= 60:
        raise ValueError("Minutos fuera de rango [0-59]")
    if s < 0 or s >= 60:
        raise ValueError("Segundos fuera de rango [0-59.999]")
    d = dms_to_decimal(g, m, s)
    if d < -180 or d > 180:
        raise ValueError("Longitud fuera de rango [-180°, 180°]")
    return True

def validar_xyz(X, Y, Z):
    """Valida coordenadas cartesianas"""
    dist = np.sqrt(X**2 + Y**2 + Z**2)
    if dist < 6306752 or dist > 6428137:
        raise ValueError(f"Coordenadas irreales. Distancia: {dist/1000:.2f} km (rango: 6,306-6,428 km)")
    return True

# ============================================
# TRANSFORMACIONES DE LATITUD
# ============================================
def phi_to_theta(phi):
    """φ → θ"""
    return np.arctan((1 - F) * np.tan(phi))

def phi_to_omega(phi):
    """φ → ω"""
    return np.arctan((1 - E2) * np.tan(phi))

def theta_to_phi(theta):
    """θ → φ"""
    return np.arctan(np.tan(theta) / (1 - F))

def omega_to_phi(omega):
    """ω → φ"""
    return np.arctan(np.tan(omega) / (1 - E2))

# ============================================
# TRANSFORMACIONES GEODÉSICAS
# ============================================
def phi_lambda_to_xyz(phi_deg, lambda_deg, h=0):
    """(φ, λ, h) → (X, Y, Z)"""
    phi = np.radians(phi_deg)
    lam = np.radians(lambda_deg)
    N = A / np.sqrt(1 - E2 * np.sin(phi)**2)
    X = (N + h) * np.cos(phi) * np.cos(lam)
    Y = (N + h) * np.cos(phi) * np.sin(lam)
    Z = (N * (1 - E2) + h) * np.sin(phi)
    return float(X), float(Y), float(Z)

def xyz_to_phi_lambda_h(X, Y, Z):
    """(X, Y, Z) → (φ, λ, h) - Algoritmo iterativo de Bowring"""
    validar_xyz(X, Y, Z)
    
    lam = np.arctan2(Y, X)
    p = np.sqrt(X**2 + Y**2)
    phi = np.arctan2(Z, p * (1 - E2))
    
    # Iteración de Bowring (10 iteraciones para convergencia)
    for _ in range(10):
        N = A / np.sqrt(1 - E2 * np.sin(phi)**2)
        h = p / np.cos(phi) - N
        phi = np.arctan2(Z, p * (1 - E2 * N / (N + h)))
    
    return np.degrees(phi), np.degrees(lam), h

# ============================================
# ARCOS GEODÉSICOS
# ============================================
def arco_meridiano(phi1, phi2, lam):
    """Calcula longitud de arco de meridiano"""
    phi1_r, phi2_r = np.radians(phi1), np.radians(phi2)
    lam_r = np.radians(lam)
    
    # Coeficientes de la serie de Fourier
    A_calc = A * (1 - E2)
    e4, e6 = E2**2, E2**3
    c0 = 1 + 3*E2/4 + 45*e4/64 + 175*e6/256
    c2 = 3*E2/4 + 15*e4/16 + 525*e6/512
    c4 = 15*e4/64 + 105*e6/256
    c6 = 35*e6/512
    
    # Integral de arco meridiano
    M = lambda phi: A_calc * (c0*phi - c2*np.sin(2*phi)/2 + c4*np.sin(4*phi)/4 - c6*np.sin(6*phi)/6)
    s = M(phi2_r) - M(phi1_r)
    
    # Puntos 3D para visualización
    puntos = []
    for phi in np.linspace(phi1_r, phi2_r, 100):
        N = A / np.sqrt(1 - E2 * np.sin(phi)**2)
        X = N * np.cos(phi) * np.cos(lam_r)
        Y = N * np.cos(phi) * np.sin(lam_r)
        Z = N * (1 - E2) * np.sin(phi)
        puntos.append([float(X), float(Y), float(Z)])
    
    return abs(s), puntos

def arco_paralelo(phi, lam1, lam2):
    """Calcula longitud de arco de paralelo"""
    phi_r = np.radians(phi)
    lam1_r, lam2_r = np.radians(lam1), np.radians(lam2)
    N = A / np.sqrt(1 - E2 * np.sin(phi_r)**2)
    r = N * np.cos(phi_r)
    s = r * abs(lam2_r - lam1_r)
    
    # Puntos 3D para visualización
    puntos = []
    for lam in np.linspace(lam1_r, lam2_r, 100):
        X = N * np.cos(phi_r) * np.cos(lam)
        Y = N * np.cos(phi_r) * np.sin(lam)
        Z = N * (1 - E2) * np.sin(phi_r)
        puntos.append([float(X), float(Y), float(Z)])
    
    return s, puntos

def biseccion_geodesica(norte_a, este_a, norte_b, este_b, alpha, beta, max_iter=100, tol=1e-6):
    """
    Método de Bisección Geodésica (Pothenot)
    
    Parámetros:
        norte_a, este_a: Coordenadas del punto A (conocido)
        norte_b, este_b: Coordenadas del punto B (conocido)
        alpha: Ángulo desde A hacia P (grados decimales, desde el norte horario)
        beta: Ángulo desde B hacia P (grados decimales, desde el norte horario)
        max_iter: Máximo número de iteraciones
        tol: Tolerancia de convergencia (metros)
    
    Retorna:
        norte_p, este_p, iteraciones, error
    """
    
    # Convertir ángulos a radianes
    alpha_rad = np.radians(alpha)
    beta_rad = np.radians(beta)
    
    # Calcular azimut de A a B
    delta_este = este_b - este_a
    delta_norte = norte_b - norte_a
    azimut_ab = np.arctan2(delta_este, delta_norte)
    dist_ab = np.sqrt(delta_norte**2 + delta_este**2)
    
    # Validar que los ángulos no sean colineales
    diff_angulos = abs(alpha - beta)
    if diff_angulos < 1 or abs(diff_angulos - 180) < 1:
        raise ValueError("Los ángulos α y β no pueden ser casi iguales o suplementarios (solución indeterminada)")
    
    # Método de la cotangente (fórmula de Collins)
    # Este es el método analítico para bisección geodésica
    
    # Convertir ángulos a sistema local
    # Ángulo α respecto a la línea AB
    gamma_a = alpha_rad - azimut_ab
    
    # Ángulo β respecto a la línea BA
    azimut_ba = azimut_ab + np.pi
    gamma_b = beta_rad - azimut_ba
    
    # Normalizar ángulos
    while gamma_a < -np.pi: gamma_a += 2 * np.pi
    while gamma_a > np.pi: gamma_a -= 2 * np.pi
    while gamma_b < -np.pi: gamma_b += 2 * np.pi
    while gamma_b > np.pi: gamma_b -= 2 * np.pi
    
    # Validar geometría
    if abs(gamma_a) < 0.01 or abs(gamma_b) < 0.01:
        raise ValueError("Configuración geométrica inválida: ángulos muy pequeños")
    
    if abs(gamma_a + gamma_b) < 0.01:
        raise ValueError("Configuración geométrica inválida: punto P en la línea AB")
    
    # Fórmula de bisección usando cotangentes
    cot_gamma_a = 1 / np.tan(gamma_a)
    cot_gamma_b = 1 / np.tan(gamma_b)
    
    # Calcular coordenadas de P usando el método de Collins
    # P está en la intersección de dos líneas desde A y B
    
    # Sistema de ecuaciones:
    # Este_P = Este_A + t * sin(alpha_rad)
    # Norte_P = Norte_A + t * cos(alpha_rad)
    # Este_P = Este_B + s * sin(beta_rad)
    # Norte_P = Norte_B + s * cos(beta_rad)
    
    # Resolver sistema
    sin_alpha = np.sin(alpha_rad)
    cos_alpha = np.cos(alpha_rad)
    sin_beta = np.sin(beta_rad)
    cos_beta = np.cos(beta_rad)
    
    # Determinante
    det = sin_alpha * cos_beta - cos_alpha * sin_beta
    
    if abs(det) < 1e-10:
        raise ValueError("Las líneas son paralelas o casi paralelas (solución indeterminada)")
    
    # Distancia desde A hasta P (parámetro t)
    delta_n = norte_b - norte_a
    delta_e = este_b - este_a
    
    t = (delta_e * cos_beta - delta_n * sin_beta) / det
    
    if t < 0:
        raise ValueError("Solución geométricamente inválida (t < 0): revise los ángulos")
    
    # Calcular coordenadas de P
    norte_p = norte_a + t * cos_alpha
    este_p = este_a + t * sin_alpha
    
    # Verificación: calcular desde B también
    s = (delta_e * cos_alpha - delta_n * sin_alpha) / det
    
    norte_p_check = norte_b + s * cos_beta
    este_p_check = este_b + s * sin_beta
    
    # Error de verificación
    error = np.sqrt((norte_p - norte_p_check)**2 + (este_p - este_p_check)**2)
    
    if error > 1.0:
        raise ValueError(f"Error de convergencia alto ({error:.3f} m): revise los datos de entrada")
    
    # Número de iteraciones (para este método directo es 1)
    iteraciones = 1
    
    return norte_p, este_p, iteraciones, error

def area_cuadrilatero(coords):
    """
    Calcula el área de un cuadrilátero sobre el elipsoide
    coords: lista de 4 tuplas [(φ1,λ1), (φ2,λ2), (φ3,λ3), (φ4,λ4)]
    Método: Proyección estereográfica local + Shoelace formula
    Los puntos se calculan SOBRE LA SUPERFICIE (h=0)
    """
    # Convertir a radianes
    coords_rad = [(np.radians(phi), np.radians(lam)) for phi, lam in coords]
    
    # Punto central para proyección
    phi_c = np.mean([c[0] for c in coords_rad])
    lam_c = np.mean([c[1] for c in coords_rad])
    
    # Proyección estereográfica local
    def proyectar(phi, lam):
        N = A / np.sqrt(1 - E2 * np.sin(phi)**2)
        M = A * (1 - E2) / (1 - E2 * np.sin(phi)**2)**(3/2)
        dphi = phi - phi_c
        dlam = lam - lam_c
        x = M * dphi
        y = N * np.cos(phi) * dlam
        return x, y
    
    # Proyectar puntos
    puntos_2d = [proyectar(phi, lam) for phi, lam in coords_rad]
    
    # Fórmula del cordón de zapato (Shoelace formula)
    area = 0
    n = len(puntos_2d)
    for i in range(n):
        j = (i + 1) % n
        area += puntos_2d[i][0] * puntos_2d[j][1]
        area -= puntos_2d[j][0] * puntos_2d[i][1]
    area = abs(area) / 2.0
    
    # Generar puntos 3D SOBRE LA SUPERFICIE (h=0)
    puntos_3d = []
    for phi, lam in coords_rad:
        X, Y, Z = phi_lambda_to_xyz(np.degrees(phi), np.degrees(lam), h=0)
        puntos_3d.append([float(X), float(Y), float(Z)])
    
    return area, puntos_3d

# ============================================
# RUTAS
# ============================================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/apartado/<int:numero>')
def apartado(numero):
    return render_template(f'apartado{numero}.html')

# ============================================
# API ENDPOINTS
# ============================================

@app.route('/api/apartado1', methods=['POST'])
def api_apartado1():
    """Transformación de latitudes (φ, θ, ω) con longitud para posición 3D completa"""
    try:
        data = request.json
        tipo = data['tipo_entrada']
        lat_g, lat_m, lat_s = float(data['lat_grados']), float(data['lat_minutos']), float(data['lat_segundos'])
        lon_g, lon_m, lon_s = float(data['lon_grados']), float(data['lon_minutos']), float(data['lon_segundos'])
        
        validar_latitud(lat_g, lat_m, lat_s)
        validar_longitud(lon_g, lon_m, lon_s)
        
        lat_decimal = dms_to_decimal(lat_g, lat_m, lat_s)
        lon_decimal = dms_to_decimal(lon_g, lon_m, lon_s)
        
        val = np.radians(lat_decimal)
        
        if tipo == 'phi':
            phi_r, theta_r, omega_r = val, phi_to_theta(val), phi_to_omega(val)
        elif tipo == 'theta':
            theta_r = val
            phi_r = theta_to_phi(val)
            omega_r = phi_to_omega(phi_r)
        else:  # omega
            omega_r = val
            phi_r = omega_to_phi(val)
            theta_r = phi_to_theta(phi_r)
        
        phi_deg, theta_deg, omega_deg = np.degrees(phi_r), np.degrees(theta_r), np.degrees(omega_r)
        X, Y, Z = phi_lambda_to_xyz(phi_deg, lon_decimal, h=0)
        
        return jsonify({
            'success': True,
            'resultados': {
                'phi': {'decimal': phi_deg, 'dms': decimal_to_dms(phi_deg)},
                'theta': {'decimal': theta_deg, 'dms': decimal_to_dms(theta_deg)},
                'omega': {'decimal': omega_deg, 'dms': decimal_to_dms(omega_deg)},
                'coords': {'X': X, 'Y': Y, 'Z': Z},
                'longitud': {'decimal': lon_decimal, 'dms': decimal_to_dms(lon_decimal)}
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/apartado2', methods=['POST'])
def api_apartado2():
    """Coordenadas con longitud (θ,λ), (φ,λ), (ω,λ) → (X,Y,Z)"""
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
    """
    Coordenadas con altura (φ,λ,h) → (X,Y,Z)
    Y problema inverso (X,Y,Z) → (φ,λ,h)
    """
    try:
        data = request.json
        
        # Detectar si es problema directo o inverso
        if 'X' in data and 'Y' in data and 'Z' in data:
            # PROBLEMA INVERSO: XYZ → φλh
            X, Y, Z = float(data['X']), float(data['Y']), float(data['Z'])
            phi, lam, h = xyz_to_phi_lambda_h(X, Y, Z)
            
            return jsonify({
                'success': True,
                'tipo': 'inverso',
                'phi': {'decimal': phi, 'dms': decimal_to_dms(phi)},
                'lambda': {'decimal': lam, 'dms': decimal_to_dms(lam)},
                'h': h,
                'X': X, 'Y': Y, 'Z': Z
            })
        else:
            # PROBLEMA DIRECTO: φλh → XYZ
            phi_g, phi_m, phi_s = float(data['phi_grados']), float(data['phi_minutos']), float(data['phi_segundos'])
            lam_g, lam_m, lam_s = float(data['lambda_grados']), float(data['lambda_minutos']), float(data['lambda_segundos'])
            h = float(data['h'])
            
            validar_latitud(phi_g, phi_m, phi_s)
            validar_longitud(lam_g, lam_m, lam_s)
            
            phi = dms_to_decimal(phi_g, phi_m, phi_s)
            lam = dms_to_decimal(lam_g, lam_m, lam_s)
            X, Y, Z = phi_lambda_to_xyz(phi, lam, h)
            
            return jsonify({
                'success': True,
                'tipo': 'directo',
                'X': X, 'Y': Y, 'Z': Z
            })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/apartado4', methods=['POST'])
def api_apartado4():
    """
    Coordenadas con h=0, conocida φ/θ/ω y λ → (X,Y,Z)
    """
    try:
        data = request.json
        tipo = data['tipo_entrada']
        lat_g, lat_m, lat_s = float(data['lat_grados']), float(data['lat_minutos']), float(data['lat_segundos'])
        lon_g, lon_m, lon_s = float(data['lon_grados']), float(data['lon_minutos']), float(data['lon_segundos'])
        
        validar_latitud(lat_g, lat_m, lat_s)
        validar_longitud(lon_g, lon_m, lon_s)
        
        lat = dms_to_decimal(lat_g, lat_m, lat_s)
        lon = dms_to_decimal(lon_g, lon_m, lon_s)
        
        # Transformar según el tipo de latitud
        if tipo == 'theta':
            lat = np.degrees(theta_to_phi(np.radians(lat)))
        elif tipo == 'omega':
            lat = np.degrees(omega_to_phi(np.radians(lat)))
        
        # Calcular con h=0
        X, Y, Z = phi_lambda_to_xyz(lat, lon, h=0)
        
        return jsonify({
            'success': True,
            'X': X, 'Y': Y, 'Z': Z,
            'lat_geodesica': lat,
            'tipo_original': tipo
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/apartado5', methods=['POST'])
def api_apartado5():
    """Arco de meridiano"""
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
    """Arco de paralelo"""
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
    """Nivelación diferencial"""
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
    """
    Bisección Geodésica (Problema de Pothenot)
    Determina coordenadas de punto P desconocido mediante ángulos desde puntos A y B conocidos
    """
    try:
        data = request.json
        norte_a = float(data['norte_a'])
        este_a = float(data['este_a'])
        norte_b = float(data['norte_b'])
        este_b = float(data['este_b'])
        
        alpha_g = float(data['alpha_grados'])
        alpha_m = float(data['alpha_minutos'])
        alpha_s = float(data['alpha_segundos'])
        beta_g = float(data['beta_grados'])
        beta_m = float(data['beta_minutos'])
        beta_s = float(data['beta_segundos'])
        
        # Validación de minutos y segundos
        if alpha_m < 0 or alpha_m >= 60 or beta_m < 0 or beta_m >= 60:
            raise ValueError("Minutos fuera de rango [0-59]")
        if alpha_s < 0 or alpha_s >= 60 or beta_s < 0 or beta_s >= 60:
            raise ValueError("Segundos fuera de rango [0-59.999]")
        
        # Convertir a decimal
        alpha = dms_to_decimal(alpha_g, alpha_m, alpha_s)
        beta = dms_to_decimal(beta_g, beta_m, beta_s)
        
        # Validar rango de ángulos
        if alpha < 0 or alpha >= 360:
            raise ValueError("Ángulo α fuera de rango [0°, 360°)")
        if beta < 0 or beta >= 360:
            raise ValueError("Ángulo β fuera de rango [0°, 360°)")
        
        # Verificar que A y B no sean el mismo punto
        if abs(norte_a - norte_b) < 1e-6 and abs(este_a - este_b) < 1e-6:
            raise ValueError("Los puntos A y B deben ser diferentes")
        
        # Resolver bisección
        norte_p, este_p, iteraciones, error = biseccion_geodesica(
            norte_a, este_a, norte_b, este_b, alpha, beta
        )
        
        return jsonify({
            'success': True,
            'punto_p': {
                'norte': float(norte_p),
                'este': float(este_p)
            },
            'iteraciones': iteraciones,
            'error': float(error)
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})
def api_apartado9():
    """Área de cuadrilátero en elipsoide"""
    try:
        data = request.json
        coords = []
        
        for i in range(1, 5):
            phi_g = float(data[f'phi{i}_grados'])
            phi_m = float(data.get(f'phi{i}_minutos', 0))
            phi_s = float(data.get(f'phi{i}_segundos', 0))
            lam_g = float(data[f'lambda{i}_grados'])
            lam_m = float(data.get(f'lambda{i}_minutos', 0))
            lam_s = float(data.get(f'lambda{i}_segundos', 0))
            
            validar_latitud(phi_g, phi_m, phi_s)
            validar_longitud(lam_g, lam_m, lam_s)
            
            phi = dms_to_decimal(phi_g, phi_m, phi_s)
            lam = dms_to_decimal(lam_g, lam_m, lam_s)
            coords.append((phi, lam))
        
        area, puntos = area_cuadrilatero(coords)
        
        return jsonify({
            'success': True,
            'area': area,
            'area_km2': area / 1e6,
            'area_ha': area / 1e4,
            'puntos': puntos,
            'coordenadas': coords
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)