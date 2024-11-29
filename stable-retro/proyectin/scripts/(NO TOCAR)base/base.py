import retro
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')  # Forzar el uso de un backend que no necesita GUI (sin Qt)

# Mapeo de botones basado en la tabla proporcionada
botones = [
    "B", "Y", "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A", "X", "L", "R"
]

def describir_accion(action):
    """Convierte el vector de acción binario en una lista de botones presionados."""
    descripcion = []
    for i, estado in enumerate(action):
        if estado == 1:
            descripcion.append(botones[i])
    return descripcion

def main():
    # Especificar el juego
    game = "GradiusIII-Snes"
    env = retro.make(game)

    # Mostrar las acciones disponibles
    print("Acciones disponibles:", env.action_space)

    done = False
    obs = env.reset()
    total_reward = 0
    step_count = 0  # Contador de pasos local (sin reinicios)
    global_step = 0  # Contador de pasos global (sin reinicios por vidas)
    max_lives = 3   # Número de vidas permitidas antes de reiniciar el entorno

    # Listas para almacenar puntajes y pasos
    scores = []
    steps = []

    # Bucle principal del juego donde el agente controla el juego
    while True:
        done = False
        obs = env.reset()
        step_count = 0
        global_step = 0
        total_reward = 0

        print("Inicio del juego")

        # Bucle del juego por intento
        while not done:
            action = np.zeros(12, dtype=int)  # Inicia el vector de acción con todos los botones sin presionar
            
            # Control de tiempo para mover arriba y abajo
            if step_count < 100:  # Moverse hacia arriba durante los primeros 100 pasos
                action[4] = 1  # Mover hacia arriba (UP)
            elif step_count < 200:  # Moverse hacia abajo durante los siguientes 100 pasos
                action[5] = 1  # Mover hacia abajo (DOWN)
            else:  # Después de los 200 pasos, reiniciar el ciclo
                step_count = 0

            # Disparar constantemente durante todo el tiempo
            action[0] = 1  # Disparar (B)
            
            # Ejecutar la acción en el entorno
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1  # Aumenta el contador de pasos (local)
            global_step += 1  # Aumenta el contador de pasos global (sin reinicios)

            # Obtener puntaje y vidas del juego
            score = info.get('score', 0)  # Puntaje actual
            lives = info.get('lives', 0)  # Número de vidas actuales
            max_score = max(scores, default=0)  # Puntaje máximo alcanzado hasta el momento
            print(f"Paso global: {global_step} - Paso local: {step_count} - Puntaje máximo: {max_score} - Puntaje: {score} - Vidas: {lives}")

            # Registrar el puntaje y el número de pasos globales
            scores.append(score)
            steps.append(global_step)  # Usar global_step en lugar de step_count

            # Si el jugador pierde todas sus vidas, reiniciar el entorno
            if lives == 0:
                print("¡Perdiste todas las vidas! Fin del juego.")
                done = True  # Finalizar el juego

            # Verificar si el juego ha terminado
            if done or truncated:
                print(f"Juego terminado. Puntaje total: {total_reward}")
                break  # Salir del bucle de juego actual

        # Verificar si se ha alcanzado el número máximo de intentos
        print("Juego finalizado.")
        break  # Salir del bucle principal después de que se termine el juego

    # Guardar el gráfico de puntaje vs pasos
    plt.figure(figsize=(10, 6))
    plt.plot(steps, scores, label="Puntaje")
    plt.xlabel('Pasos')
    plt.ylabel('Puntaje')
    plt.title('Puntaje vs Pasos')
    plt.grid(True)
    plt.legend()

    # Guardar el gráfico como archivo
    plt.savefig('score_vs_steps.png')  # Guarda el gráfico como un archivo PNG
    print("Gráfico guardado como 'score_vs_steps.png'.")

    env.close()  # Cerrar el entorno cuando haya terminado

if __name__ == "__main__":
    main()
