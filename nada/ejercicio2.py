def vocal_frecuente(texto):
    frecuencia_vocales = {'a': 0, 'e': 0, 'i': 0, 'o': 0, 'u': 0}
    
    # Convertir el texto a minúsculas
    texto = texto.lower()
    
    # Recorrer el texto y contar la frecuencia de las vocales sin tildes
    for caracter in texto:
        if caracter in frecuencia_vocales:
            frecuencia_vocales[caracter] += 1
    
    # Encontrar la frecuencia máxima
    max_frecuencia = max(frecuencia_vocales.values())
   

    vocales_frecuentes = []
    for letra in frecuencia_vocales:
        repeticion = frecuencia_vocales.get(letra)
        if repeticion > 1:
            vocales_frecuentes.append(letra)

    if max_frecuencia == 0:
        return 'No hay vocales en el texto'
    else:
        return ', '.join(vocales_frecuentes)

# Ejemplo de uso:
texto = input("Ingresa una frase: ")

resultado = vocal_frecuente(texto)

print(f"Frese: {texto} - Resultado: {resultado}")
