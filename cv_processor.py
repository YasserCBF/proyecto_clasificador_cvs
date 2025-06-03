import re
from PyPDF2 import PdfReader
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.util import bigrams, trigrams
from collections import Counter
from math import log

# Cachear stopwords para múltiples idiomas
STOP_WORDS = {
    'es': set(stopwords.words('spanish') + list(string.punctuation)),
    'en': set(stopwords.words('english') + list(string.punctuation)),
    'fr': set(stopwords.words('french') + list(string.punctuation))
}

# Diccionario de palabras clave ampliado con más categorías
KEYWORDS = {
    'Seguridad y Limpieza': {
        'words': {'limpieza': 1.0, 'seguridad': 1.5, 'mantenimiento': 1.2, 'higiene': 1.0, 'vigilancia': 0.8, 'control': 0.8, 'supervisión': 0.8, 'desinfección': 0.7},
        'bigrams': {'seguridad laboral': 2.0, 'control calidad': 1.8, 'mantenimiento equipos': 1.5},
        'trigrams': {'seguridad en trabajo': 2.5, 'control de calidad': 2.0, 'higiene en workplace': 1.8},
        'synonyms': {'aseo': 0.5, 'protección': 0.5, 'saneamiento': 0.5}
    },
    'Marketing': {
        'words': {'publicidad': 1.2, 'ventas': 1.0, 'estrategia': 0.8, 'mercado': 1.0, 'promoción': 1.2, 'análisis': 0.7, 'branding': 0.9, 'clientes': 0.6},
        'bigrams': {'estrategia mercado': 2.0, 'análisis mercado': 1.8, 'campaña publicidad': 2.0},
        'trigrams': {'estrategia de ventas': 2.5, 'análisis de mercado': 2.0, 'promoción de marca': 1.8},
        'synonyms': {'mercadotecnia': 0.5, 'promocional': 0.5, 'comercialización': 0.5}
    },
    'Finanzas': {
        'words': {'contabilidad': 1.5, 'presupuesto': 1.2, 'finanzas': 1.5, 'inversión': 1.0, 'tesorería': 1.2, 'auditoría': 1.2, 'ingresos': 0.7, 'egresos': 0.7, 'costos': 0.6},
        'bigrams': {'gestión financiera': 2.0, 'análisis financiero': 1.8, 'auditoría cuentas': 1.5},
        'trigrams': {'gestión de presupuestos': 2.5, 'análisis de costos': 2.0, 'auditoría de finanzas': 1.8},
        'synonyms': {'contaduría': 0.5, 'balance': 0.5, 'económico': 0.5}
    },
    'Tecnología': {
        'words': {'programación': 1.2, 'desarrollo': 1.0, 'software': 1.0, 'sistemas': 0.8, 'algoritmos': 0.9, 'base de datos': 1.0, 'código': 0.7, 'python': 0.6, 'java': 0.6, 'html': 0.5},
        'bigrams': {'desarrollo software': 2.0, 'base datos': 1.8, 'sistemas operativos': 1.5},
        'trigrams': {'desarrollo de software': 2.5, 'base de datos': 2.0, 'programación en python': 1.8},
        'synonyms': {'informática': 0.5, 'tecnología': 0.5, 'codificación': 0.5}
    },
    'Atención al Cliente': {
        'words': {'servicio': 1.0, 'cliente': 1.0, 'resolución': 0.8, 'soporte': 0.9, 'comunicación': 0.7, 'atención': 1.0, 'satisfacción': 0.8, 'problemas': 0.6},
        'bigrams': {'atención cliente': 2.0, 'resolución problemas': 1.8, 'soporte técnico': 1.5},
        'trigrams': {'atención al cliente': 2.5, 'resolución de problemas': 2.0, 'soporte al cliente': 1.8},
        'synonyms': {'asistencia': 0.5, 'ayuda': 0.5, 'servicio al cliente': 0.5}
    },
    'Recursos Humanos': {
        'words': {'reclutamiento': 1.2, 'selección': 1.0, 'capacitación': 1.0, 'desarrollo personal': 0.9, 'beneficios': 0.8, 'nómina': 0.7, 'evaluación': 0.6},
        'bigrams': {'reclutamiento personal': 2.0, 'capacitación empleados': 1.8, 'evaluación desempeño': 1.5},
        'trigrams': {'desarrollo de personal': 2.5, 'evaluación de desempeño': 2.0, 'gestión de nómina': 1.8},
        'synonyms': {'recursos humanos': 0.5, 'talento': 0.5, 'rrhh': 0.5}
    },
    'Ventas': {
        'words': {'ventas': 1.5, 'negociación': 1.2, 'clientes': 1.0, 'prospectos': 0.9, 'cierre': 0.8, 'comercial': 0.7, 'cuotas': 0.6},
        'bigrams': {'ventas directas': 2.0, 'negociación clientes': 1.8, 'cierre deals': 1.5},
        'trigrams': {'ventas de productos': 2.5, 'negociación de contratos': 2.0, 'gestión de clientes': 1.8},
        'synonyms': {'comercio': 0.5, 'vendedor': 0.5, 'ventas online': 0.5}
    }
}

# Palabras ambiguas que penalizan
AMBIGUOUS_WORDS = {'gestión': 0.3, 'análisis': 0.2, 'coordinación': 0.2, 'planificación': 0.2, 'desarrollo': 0.1}

# Diccionario de lematización ampliado
LEMMAS = {
    'contabilidades': 'contabilidad', 'finanzas': 'finanza', 'presupuestos': 'presupuesto',
    'publicidades': 'publicidad', 'ventas': 'venta', 'estrategias': 'estrategia',
    'limpiezas': 'limpieza', 'seguridades': 'seguridad', 'mantenimientos': 'mantenimiento',
    'desarrollos': 'desarrollo', 'programaciones': 'programación', 'clientes': 'cliente',
    'servicios': 'servicio', 'satisfacciones': 'satisfacción', 'resoluciones': 'resolución',
    'inversiones': 'inversión', 'auditorías': 'auditoría', 'promociones': 'promoción',
    'bases de datos': 'base de datos', 'codigos': 'código', 'soporte técnico': 'soporte',
    'atenciones': 'atención', 'comunicaciones': 'comunicación', 'costos': 'costo',
    'ingresos': 'ingreso', 'egresos': 'egreso', 'análisis': 'análisis',
    'reclutamientos': 'reclutamiento', 'selecciones': 'selección', 'capacitaciones': 'capacitación',
    'negociaciones': 'negociación', 'prospectos': 'prospecto', 'cierres': 'cierre'
}

# Diccionario de palabras para análisis de sentimientos
SENTIMENT_WORDS = {
    'positive': {'excelente', 'satisfactorio', 'exitoso', 'positivo', 'beneficioso', 'eficaz', 'productivo', 'grande'},
    'negative': {'difícil', 'fallo', 'problema', 'error', 'fracaso', 'malo', 'pésimo', 'insatisfactorio'},
    'neutral': {'trabajo', 'experiencia', 'habilidad', 'conocimiento', 'formación', 'estudio'}
}

def detect_language(text):
    """Detecta el idioma basado en caracteres y palabras clave"""
    spanish_chars = set('áéíóúñÁÉÍÓÚÑ')
    english_chars = set('qwertyuiopasdfghjklzxcvbnm')
    french_chars = set('àâçéèêëîïôùûüÿÀÂÇÉÈÊËÎÏÔÙÛÜŸ')
    
    spanish_count = sum(1 for char in text if char in spanish_chars)
    english_count = sum(1 for char in text if char in english_chars and char not in spanish_chars)
    french_count = sum(1 for char in text if char in french_chars)
    
    if spanish_count > max(english_count, french_count) and spanish_count > 5:
        return 'es'
    elif english_count > max(spanish_count, french_count) and len(text.split()) > 10:
        return 'en'
    elif french_count > max(spanish_count, english_count) and len(text.split()) > 10:
        return 'fr'
    return 'es'  # Por defecto español si no se detecta claramente

def extract_text(file_path):
    try:
        if file_path.endswith('.pdf'):
            reader = PdfReader(file_path)
            text = ''.join([page.extract_text() or '' for page in reader.pages])
        elif file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            raise ValueError("Tipo de archivo no soportado.")
        
        if not text.strip():
            raise ValueError("El archivo está vacío o no contiene texto legible.")
        
        lang = detect_language(text)
        text = text.lower()
        tokens = word_tokenize(text)
        tokens = [LEMMAS.get(token, token) for token in tokens if token.isalpha() and token not in STOP_WORDS[lang]]
        return ' '.join(tokens), tokens, lang
    except Exception as e:
        raise RuntimeError(f"Error extrayendo texto: {str(e)}")

def identify_sections(text):
    sections = {'experiencia': [], 'habilidades': [], 'educacion': [], 'resumen': [], 'general': []}
    current_section = 'general'
    section_patterns = {
        'experiencia': r'\b(experiencia|laboral|empleo|historial\s+(laboral|profesional)|trabajo)\b',
        'habilidades': r'\b(habilidades|competencias|aptitudes|conocimientos|destrezas)\b',
        'educacion': r'\b(educación|formación|estudios|académica|académico)\b',
        'resumen': r'\b(resumen|perfil|objetivo|acerca\s+de\s+mí|sobre\s+mí)\b'
    }
    
    for line in text.split('\n'):
        line = line.strip().lower()
        found_section = False
        for section, pattern in section_patterns.items():
            if re.search(pattern, line):
                current_section = section
                found_section = True
                break
        if line:
            sections[current_section].append(line)
    
    return sections

def detect_negations(words, keyword_index, window=3):
    negations = {'no', 'nunca', 'jamás', 'ni', 'tampoco', 'not', 'ne', 'pas'}
    start = max(0, keyword_index - window)
    end = min(len(words), keyword_index + window + 1)
    nearby_words = words[start:end]
    return any(word in negations for word in nearby_words)

def analyze_sentiment(tokens):
    sentiment_scores = {'positive': 0, 'negative': 0, 'neutral': 0}
    for token in tokens:
        if token in SENTIMENT_WORDS['positive']:
            sentiment_scores['positive'] += 1
        elif token in SENTIMENT_WORDS['negative']:
            sentiment_scores['negative'] += 1
        elif token in SENTIMENT_WORDS['neutral']:
            sentiment_scores['neutral'] += 1
    total = sum(sentiment_scores.values())
    if total == 0:
        return 'neutral'
    if sentiment_scores['positive'] > sentiment_scores['negative'] and sentiment_scores['positive'] > sentiment_scores['neutral']:
        return 'positive'
    elif sentiment_scores['negative'] > sentiment_scores['positive'] and sentiment_scores['negative'] > sentiment_scores['neutral']:
        return 'negative'
    return 'neutral'

def classify_cv(file_path, return_stats=False):
    text, tokens, lang = extract_text(file_path)
    
    # Validar calidad del texto
    if len(tokens) < 30:
        result = "No clasificado: Contenido insuficiente para clasificar."
        return (result, {}) if return_stats else result
    
    # Calcular frecuencia inversa (TF-IDF simplificado)
    token_counts = Counter(tokens)
    total_tokens = len(tokens)
    inverse_freq = {token: log(total_tokens / (1 + count)) for token, count in token_counts.items()}
    
    # Generar bigramas y trigramas
    text_bigrams = [' '.join(bg) for bg in bigrams(tokens)]
    text_trigrams = [' '.join(tg) for tg in trigrams(tokens)]
    
    # Identificar secciones
    sections = identify_sections(text)
    
    # Procesar por secciones
    section_tokens = {section: word_tokenize(' '.join(content)) for section, content in sections.items()}
    section_bigrams = {section: [' '.join(bg) for bg in bigrams(st)] for section, st in section_tokens.items()}
    section_trigrams = {section: [' '.join(tg) for tg in trigrams(st)] for section, st in section_tokens.items()}
    
    # Calcular densidad de palabras clave
    keyword_count = 0
    scores = {area: {'total': 0.0, 'details': [], 'keyword_count': 0} for area in KEYWORDS}
    section_weights = {'experiencia': 2.0, 'habilidades': 1.5, 'educacion': 0.8, 'resumen': 0.5, 'general': 0.3}
    
    # Procesar cada área según el idioma detectado
    keyword_dict = KEYWORDS if lang == 'es' else {k: v for k, v in KEYWORDS.items() if any(w in v['words'] for w in tokens)}  # Filtrar por idioma
    for area, data in keyword_dict.items():
        # Palabras clave individuales
        for section, tokens in section_tokens.items():
            for i, word in enumerate(tokens):
                if word in data['words']:
                    if detect_negations(tokens, i):
                        scores[area]['details'].append(f"'{word}' en {section} (descartado por negación)")
                        continue
                    nearby_boost = sum(1 for j in range(max(0, i-5), min(len(tokens), i+6)) if tokens[j] in data['words'])
                    weight = data['words'][word] * inverse_freq.get(word, 1.0) * (1 + 0.1 * nearby_boost) * section_weights[section]
                    scores[area]['details'].append(f"'{word}' en {section} (peso {weight:.2f})")
                    scores[area]['total'] += weight
                    scores[area]['keyword_count'] += 1
                    keyword_count += 1
        
        # Bigramas
        for bigram, weight in data['bigrams'].items():
            for section, bigrams_list in section_bigrams.items():
                matches = bigrams_list.count(bigram)
                if matches > 0:
                    adjusted_weight = weight * section_weights[section]
                    scores[area]['details'].append(f"'{bigram}' en {section} ({matches} veces, peso {adjusted_weight:.2f})")
                    scores[area]['total'] += matches * adjusted_weight
                    scores[area]['keyword_count'] += matches
                    keyword_count += matches
        
        # Trigramas
        for trigram, weight in data['trigrams'].items():
            for section, trigrams_list in section_trigrams.items():
                matches = trigrams_list.count(trigram)
                if matches > 0:
                    adjusted_weight = weight * section_weights[section]
                    scores[area]['details'].append(f"'{trigram}' en {section} ({matches} veces, peso {adjusted_weight:.2f})")
                    scores[area]['total'] += matches * adjusted_weight
                    scores[area]['keyword_count'] += matches
                    keyword_count += matches
        
        # Sinónimos (solo si el puntaje es bajo)
        if scores[area]['total'] < 2.0:
            for synonym, weight in data['synonyms'].items():
                for section, tokens in section_tokens.items():
                    matches = tokens.count(synonym)
                    if matches > 0:
                        adjusted_weight = weight * inverse_freq.get(synonym, 1.0) * section_weights[section]
                        scores[area]['details'].append(f"'{synonym}' (sinónimo) en {section} ({matches} veces, peso {adjusted_weight:.2f})")
                        scores[area]['total'] += matches * adjusted_weight
                        scores[area]['keyword_count'] += matches
                        keyword_count += matches
    
    # Penalizar palabras ambiguas
    for section, tokens in section_tokens.items():
        for word, penalty in AMBIGUOUS_WORDS.items():
            matches = tokens.count(word)
            if matches > 0:
                adjusted_penalty = penalty * inverse_freq.get(word, 1.0) * section_weights[section]
                for area in scores:
                    scores[area]['details'].append(f"Penalización por '{word}' en {section} ({matches} veces, penalización -{adjusted_penalty:.2f})")
                    scores[area]['total'] -= matches * adjusted_penalty
    
    # Calcular densidad de palabras clave
    keyword_density = keyword_count / total_tokens if total_tokens > 0 else 0
    if keyword_density < 0.05:
        result = f"No clasificado: Densidad de palabras clave demasiado baja ({keyword_density:.2%})."
        stats = {area: f"{score['total']:.2f} ({score['keyword_count']} coincidencias)" for area, score in scores.items()}
        return (result, stats) if return_stats else result
    
    # Umbral de confianza
    max_score = max(score['total'] for score in scores.values())
    if max_score < 4.0:
        result = f"No clasificado: Insuficientes coincidencias relevantes (puntaje máximo: {max_score:.2f})."
        stats = {area: f"{score['total']:.2f} ({score['keyword_count']} coincidencias)" for area, score in scores.items()}
        return (result, stats) if return_stats else result
    
    # Determinar áreas principales
    top_areas = [area for area, score in scores.items() if score['total'] == max_score]
    
    # Calcular confianza
    total_score = sum(score['total'] for score in scores.values())
    confidence = (max_score / total_score) * 100 if total_score > 0 else 0
    
    # Análisis de sentimientos
    sentiment = analyze_sentiment(tokens)
    
    # Verificar empate cercano
    close_scores = [area for area, score in scores.items() if area not in top_areas and abs(score['total'] - max_score) < 1.0]
    if close_scores or confidence < 60:
        result = f"No clasificado: Resultado ambiguo o confianza baja (confianza: {confidence:.1f}%, tono: {sentiment})."
        stats = {area: f"{score['total']:.2f} ({score['keyword_count']} coincidencias, {', '.join(score['details'])})" for area, score in scores.items()}
        return (result, stats) if return_stats else result
    
    # Generar resultado final
    result = f"Clasificado como: {', '.join(top_areas)} (confianza: {confidence:.1f}%, tono: {sentiment})."
    stats = {area: f"{score['total']:.2f} ({score['keyword_count']} coincidencias, {', '.join(score['details'])})" for area, score in scores.items()}
    
    return (result, stats) if return_stats else result