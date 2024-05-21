"""
El objetivo de este ejercicio es crear un programa que:
1. Solicite al usuario ingresar una frase.
2. Cuente la ocurrencia de cada palabra en la frase y las almacene en un diccionario.
3. Encuentre la palabra que aparece más veces y la que aparece menos veces en la frase.
4. Muestre un resumen con todas las palabras y sus respectivas ocurrencias, y las palabras más y
menos frecuentes.
5. Requisitos adicionales:
6. El programa debe ignorar diferencias entre mayúsculas y minúsculas (por ejemplo, "Hola" y "hola"
deben contar como la misma palabra).
7. Puntuación como comas y puntos deben ser ignorados.
"""

# Función para contar las palabras en una frase
def contar_palabras(frase):
    # Lista de signos de puntuación a eliminar
    signos_puntuacion = ['.', ',', '!', '?', ';', ':', '-', '"', "'"]

    # Elimina la puntuación y convierte todo a minúsculas
    for signo in signos_puntuacion:
        frase = frase.replace(signo, '')
    frase = frase.lower()
    palabras = frase.split()

    # Crea un diccionario para contar las ocurrencias de cada palabra
    contador = {}
    for palabra in palabras:
        if palabra in contador:
            contador[palabra] += 1
        else:
            contador[palabra] = 1

    return contador

# Función para encontrar la palabra más y menos frecuente
def palabras_mas_menos_frecuente(contador):
    ocurrencias = []
    palabra_menos_frecuente = []
    palabra_mas_frecuente = []
    
    for palabra in contador:
        ocurrencia = contador.get(palabra)
        if ocurrencia > 1:
            palabra_mas_frecuente.append(palabra)
            ocurrencias.append(ocurrencia)
        else:
            palabra_menos_frecuente.append(palabra)

    return palabra_menos_frecuente, palabra_mas_frecuente, ocurrencias

    

# Solicita al usuario ingresar una frase
frase = input("Ingresa una frase: ")

# Cuenta las palabras en la frase
print("Frase: ",frase)
contador = contar_palabras(frase)
print('Contador de palabras: ', contador)


# Encuentra la palabra más y menos frecuente
palabra_menos_frecuente, palabra_mas_frecuente, ocurrencias = palabras_mas_menos_frecuente(contador)


print("Palabra mas frencuente: ", ", ".join(palabra_mas_frecuente) , " Ocurrencias: ",  ", ".join(map(str, ocurrencias)))
print("Palabra menos frecuente: ", ", ".join(palabra_menos_frecuente), " Ocurrencias: 1")

