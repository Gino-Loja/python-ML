
"""
# Ejercicio 1: Clase Círculo
# Enunciado: Crea una clase llamada Circulo que tenga un atributo radio.
# Agrega dos métodos a la clase: area que calcula y devuelve el área del círculo,
# y ccircunferencia que calula y devuelve la circunferencia del círculo.
# 1.El área de un círculo es pi multiplicado por el radio al cuadrado (A = π r²).
# π = 3,14 o π = 3,1416
# 2. La longitud de una circunferencia es igual a 2 pi por el radio.
# 3. Deben importar el módulo math
"""

import math

class Circulo:
    def __init__(self, radio):
        self.radio = radio

    def area(self):
        # Área de un círculo: π * radio al cuadrado
        return math.pi * self.radio ** 2

    def circunferencia(self):
        # Circunferencia de un círculo: 2 * π * radio
        return 2 * math.pi * self.radio


radio_del_circulo = int(input("Ingrese el radio"))
mi_circulo = Circulo(radio_del_circulo)

print(f"Área del círculo: {mi_circulo.area()}")
print(f"Circunferencia del círculo: {mi_circulo.circunferencia()}")
