import os
import hashlib
from datetime import datetime, timedelta
import re

def validate_file(file):
    """Valida si el archivo es legible, no está corrupto y tiene estructura de CV"""
    file.seek(0, os.SEEK_END)
    size = file.tell()
    file.seek(0)
    if size > 10 * 1024 * 1024:  # 10MB
        return False, "Archivo demasiado grande (máximo 10MB)."
    if size == 0:
        return False, "Archivo vacío."
    
    # Leer contenido preliminar
    content = file.read().decode('utf-8', errors='ignore')
    file.seek(0)
    
    # Verificar si tiene estructura básica de CV (secciones comunes)
    cv_patterns = [
        r'\b(experiencia|laboral|empleo|historial)\b',
        r'\b(habilidades|competencias|aptitudes)\b',
        r'\b(educación|formación|estudios)\b'
    ]
    has_cv_structure = any(re.search(pattern, content.lower()) for pattern in cv_patterns)
    if not has_cv_structure:
        return False, "El archivo no parece tener la estructura de un CV."
    
    return True, ""

def clean_old_files(upload_folder, max_age_hours=24):
    """Elimina archivos antiguos y duplicados en la carpeta uploads"""
    now = datetime.now()
    file_hashes = set()
    for filename in os.listdir(upload_folder):
        file_path = os.path.join(upload_folder, filename)
        if os.path.isfile(file_path):
            # Verificar antigüedad
            file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
            if now - file_mtime > timedelta(hours=max_age_hours):
                os.remove(file_path)
                continue
            
            # Verificar duplicados
            with open(file_path, 'rb') as f:
                file_hash = hashlib.md5(f.read()).hexdigest()
            if file_hash in file_hashes:
                os.remove(file_path)
            else:
                file_hashes.add(file_hash)

def save_uploaded_file(file, upload_folder, original_name, file_hash):
    """Guarda el archivo con un nombre único basado en hash"""
    clean_old_files(upload_folder)
    
    # Validar archivo
    is_valid, error_message = validate_file(file)
    if not is_valid:
        raise ValueError(error_message)
    
    # Generar nombre único
    extension = os.path.splitext(original_name)[1].lower()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    safe_filename = f"cv_{timestamp}_{file_hash[:8]}{extension}"
    file_path = os.path.join(upload_folder, safe_filename)
    
    # Guardar archivo
    file.save(file_path)
    return file_path