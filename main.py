from flask import Flask, request, render_template
import os
import hashlib
from werkzeug.utils import secure_filename
from cv_processor import classify_cv
from file_handler import save_uploaded_file

app = Flask(__name__)

# Configuración
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf', 'txt'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB como máximo

def is_allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html', result=None, stats=None)

@app.route('/upload', methods=['POST'])
def upload_cv():
    file = request.files.get('cv_file')

    # Validaciones iniciales
    if not file or file.filename.strip() == '':
        return render_template('index.html', result="Error: No se seleccionó un archivo válido.", stats=None)

    if not is_allowed_file(file.filename):
        return render_template('index.html', result="Error: Solo se permiten archivos PDF o TXT.", stats=None)

    # Validar tamaño del archivo
    file.seek(0, os.SEEK_END)
    size = file.tell()
    file.seek(0)
    if size > MAX_FILE_SIZE:
        return render_template('index.html', result="Error: El archivo excede el tamaño máximo de 10MB.", stats=None)
    if size == 0:
        return render_template('index.html', result="Error: El archivo está vacío.", stats=None)

    # Generar hash del archivo para detectar duplicados
    file_content = file.read()
    file_hash = hashlib.md5(file_content).hexdigest()
    file.seek(0)

    # Comprobar si ya se subió un archivo idéntico
    existing_files = os.listdir(app.config['UPLOAD_FOLDER'])
    for existing_file in existing_files:
        existing_path = os.path.join(app.config['UPLOAD_FOLDER'], existing_file)
        if os.path.exists(existing_path):
            try:
                with open(existing_path, 'rb') as f:
                    existing_content = f.read()
                    if hashlib.md5(existing_content).hexdigest() == file_hash:
                        return render_template('index.html', result="Error: Este archivo ya fue subido anteriormente.", stats=None)
            except Exception:
                return render_template('index.html', result="Error: Problema al verificar duplicados.", stats=None)

    try:
        filename = secure_filename(file.filename)
        file_path = save_uploaded_file(file, app.config['UPLOAD_FOLDER'], filename, file_hash)
        classification, stats = classify_cv(file_path, return_stats=True)
        return render_template('index.html', result=classification, stats=stats)
    except ValueError as ve:
        return render_template('index.html', result=f"Error de validación: {str(ve)}", stats=None)
    except RuntimeError as re:
        return render_template('index.html', result=f"Error de procesamiento: {str(re)}", stats=None)
    except Exception as e:
        return render_template('index.html', result=f"Error inesperado: {str(e)}", stats=None)

if __name__ == '__main__':
    app.run(debug=True)