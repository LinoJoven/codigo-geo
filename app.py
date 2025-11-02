from flask import Flask, render_template, request, jsonify
import numpy as np
import json

app = Flask(__name__)

# Parámetros del elipsoide WGS84
a = 6378137.0  # Semieje mayor (metros)
b = 6356752.314245  # Semieje menor (metros)
e2 = 1 - (b**2 / a**2)  # Primera excentricidad al cuadrado

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/apartado1')
def apartado1():
    return render_template('apartado1.html')

@app.route('/apartado2')
def apartado2():
    return render_template('apartado2.html')

@app.route('/apartado3')
def apartado3():
    return render_template('apartado3.html')

@app.route('/apartado4')
def apartado4():
    return render_template('apartado4.html')

@app.route('/apartado5')
def apartado5():
    return render_template('apartado5.html')

@app.route('/apartado6')
def apartado6():
    return render_template('apartado6.html')

@app.route('/apartado7')
def apartado7():
    return render_template('apartado7.html')

@app.route('/apartado8')
def apartado8():
    return render_template('apartado8.html')

@app.route('/apartado9')
def apartado9():
    return render_template('apartado9.html')

@app.route('/calcular_latitudes', methods=['POST'])
def calcular_latitudes():
    data = request.json
    tipo_entrada = data.get('tipo')
    valor = float(data.get('valor'))
    longitud = float(data.get('longitud', 0))
    
    # Convertir a radianes
    valor_rad = np.radians(valor)
    lon_rad = np.radians(longitud)
    
    if tipo_entrada == 'geodesica':
        phi = valor_rad
        # Calcular theta (paramétrica)
        theta = np.arctan(np.sqrt(1 - e2) * np.tan(phi))
        # Calcular omega (geocéntrica)
        omega = np.arctan((1 - e2) * np.tan(phi))
        
    elif tipo_entrada == 'parametrica':
        theta = valor_rad
        # Calcular phi (geodésica)
        phi = np.arctan(np.tan(theta) / np.sqrt(1 - e2))
        # Calcular omega (geocéntrica)
        omega = np.arctan((1 - e2) * np.tan(phi))
        
    elif tipo_entrada == 'geocentrica':
        omega = valor_rad
        # Calcular phi (geodésica)
        phi = np.arctan(np.tan(omega) / (1 - e2))
        # Calcular theta (paramétrica)
        theta = np.arctan(np.sqrt(1 - e2) * np.tan(phi))
    
    # Calcular coordenadas cartesianas para el gráfico 3D
    N = a / np.sqrt(1 - e2 * np.sin(phi)**2)
    x = N * np.cos(phi) * np.cos(lon_rad)
    y = N * np.cos(phi) * np.sin(lon_rad)
    z = N * (1 - e2) * np.sin(phi)
    
    # Generar puntos del elipsoide para visualización
    u = np.linspace(0, 2 * np.pi, 50)
    v = np.linspace(-np.pi/2, np.pi/2, 50)
    x_elip = a * np.outer(np.cos(u), np.cos(v))
    y_elip = a * np.outer(np.sin(u), np.cos(v))
    z_elip = b * np.outer(np.ones(np.size(u)), np.sin(v))
    
    return jsonify({
        'phi': np.degrees(phi),
        'theta': np.degrees(theta),
        'omega': np.degrees(omega),
        'punto': {'x': x/1e6, 'y': y/1e6, 'z': z/1e6},  # En millones de metros
        'elipsoide': {
            'x': x_elip.tolist(),
            'y': y_elip.tolist(),
            'z': z_elip.tolist()
        }
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)