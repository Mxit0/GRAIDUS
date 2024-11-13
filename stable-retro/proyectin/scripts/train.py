import retro
import time

def main():
    env = retro.make(game="GradiusIII-Snes")
    env.reset()

    # Define las acciones de moverse hacia arriba y hacia abajo
    action_up = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  # Solo UP activado
    action_down = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]  # Solo DOWN activado

    start_time = time.time()  # Marca el tiempo de inicio
    interval = 1  # Intervalo en segundos

    while True:
        # Alterna la acción según el tiempo transcurrido
        if int(time.time() - start_time) % (2 * interval) < interval:
            action = action_up  # Mueve hacia arriba
        else:
            action = action_down  # Mueve hacia abajo

        observation, reward, terminated, truncated, info = env.step(action)
        env.render()

        # Reinicia el juego si ha terminado
        if terminated or truncated:
            env.reset()

        time.sleep(0.02)  # Pausa para controlar la velocidad de ejecución

    env.close()

if __name__ == "__main__":
    main()
