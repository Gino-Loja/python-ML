"""
# Ejercicio 2: Clase Cuenta Bancaria
# Enunciado: Crea una clase llamada CuentaBancaria que represente una cuenta bancaria.
# Debe tener dos atributos: titular y saldo. La clase también debe tener dos métodos:
# depositar, que debe agregar el monto depositado al saldo, y retirar,
# que debe restar el monto retirado del saldo.
"""


class CuentaBancaria:
    def __init__(self, titular, saldo=0.0):
        self.titular = titular
        self.saldo = saldo

    def depositar(self, monto):
        if monto > 0:
            self.saldo += monto
            print(f"Depósito de ${monto} realizado. Nuevo saldo: ${self.saldo}")
        else:
            print("El monto del depósito debe ser mayor que cero.")

    def retirar(self, monto):
        if 0 < monto <= self.saldo:
            self.saldo -= monto
            print(f"Retiro de ${monto} realizado. Nuevo saldo: ${self.saldo}")
        elif monto > self.saldo:
            print("Fondos insuficientes.")
        else:
            print("El monto del retiro debe ser mayor que cero.")

# Cuenta de Juan Perez :) empieza con 1000$
cuenta = CuentaBancaria("Juan Pérez", 1000.0)

while True:
    print("\nMenú: Cuenta a nombre de Juan Perez")
    print("1. Depositar")
    print("2. Retirar")
    print("3. Salir")
    
    opcion = input("Selecciona una opción: ")

    if opcion == "1":
        monto = float(input("Ingrese el monto a depositar: "))
        cuenta.depositar(monto)
    elif opcion == "2":
        monto = float(input("Ingrese el monto a retirar: "))
        cuenta.retirar(monto)
    elif opcion == "3":
        print("Saliendo del programa.")
        break
    else:
        print("Opción no válida. Por favor, seleccione una opción válida.")
