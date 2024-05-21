
"""
# Ejercicio 3: Clase Proyecto
# Enunciado: Crea una clase llamada Proyecto que tenga los atributos:
# nombre, horas_estimadas, horas_trabajadas y completo. La clase debe tener tres
métodos:
# Trabajar: que aumente las horas trabajadas en el número de horas pasadas.
# Completar: que cambie el estado del proyecto a completo.
# Estado: que muestre el nombre del proyecto, el porcentaje completado
# basado en las horas trabajadas y las horas estimadas, y si el proyecto está completo o no
"""



class Proyecto:
    def __init__(self, nombre, horas_estimadas):
        self.nombre = nombre
        self.horas_estimadas = horas_estimadas
        self.horas_trabajadas = 0
        self.completo = False

    def trabajar(self, horas):
        if not self.completo:
            self.horas_trabajadas += horas
            print(f"Se han trabajado {horas} horas mas en el proyecto {self.nombre}.")
            if self.horas_trabajadas >= self.horas_estimadas:
                self.completar()

    def completar(self):
        self.completo = True
        print(f"El proyecto {self.nombre} se ha completado.")

    def estado(self):
        porcentaje_completado = (self.horas_trabajadas / self.horas_estimadas) * 100
        estado_completo = "Completo" if self.completo else "Incompleto"
        print(f"Proyecto: {self.nombre}")
        print(f"Porcentaje completado: {porcentaje_completado:.2f}%")
        print(f"Estado: {estado_completo}")

# Ejemplo de uso
proyecto = Proyecto("Proyecto IoT", 100)


while True:
    print("\nProyecto")
    print("1. Ingresar horas")
    print("2. Salir")
    
    opcion = input("Selecciona una opción: ")

  

    if opcion == "1":
        horas = int(input("Ingresa las horas que trabajaste: "))
        if horas <= 100:
            proyecto.trabajar(horas)
            proyecto.estado()
            if proyecto.completo:
                break
        else:
            print("El proyecto dura 100 horas es imposible que pases este limite.")
            break
    elif opcion == "2":
        print("Saliendo del programa.")
        break
    else:
        print("Opción no válida. Por favor, seleccione una opción válida.")

      
