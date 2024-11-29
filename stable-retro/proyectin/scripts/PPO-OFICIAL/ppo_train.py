import retro
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
import csv
from datetime import datetime, timedelta
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import WarpFrame, ClipRewardEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback

matplotlib.use('Agg')  # Para usar un backend sin necesidad de GUI

# Crear el entorno
def make_env():
    """Crea y configura el entorno de GradiusIII-Snes."""
    env = retro.make(game="GradiusIII-Snes")
    env = WarpFrame(env)  # Preprocesamiento de las imágenes
    env = ClipRewardEnv(env)  # Recorta las recompensas
    return env

# Función para guardar los datos de entrenamiento en un archivo CSV
def save_training_data(times, scores, filename="training_data.csv"):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Time (minutes)", "Score"])
        writer.writerows(zip(times, scores))
    print(f"Datos guardados en {filename}")

# Callback para recolectar datos durante el entrenamiento
class EvalCallback(BaseCallback):
    def __init__(self, start_time, verbose=0, log_interval=1000):
        super(EvalCallback, self).__init__(verbose)
        self.max_score = -np.inf  # Puntaje máximo inicial
        self.current_score = 0  # Puntaje actual
        self.log_interval = log_interval  # Intervalo para imprimir logs
        self.start_time = start_time  # Tiempo de inicio del entrenamiento
        self.step_count = 0  # Contador de pasos
        self.times_in_minutes = []  # Almacenar tiempos en minutos
        self.scores = []  # Almacenar puntajes

    def _on_step(self) -> bool:
        """
        Este método se llama después de cada paso de entrenamiento.
        Guarda el puntaje máximo y actual en cada paso, y registra el tiempo en minutos.
        """
        self.step_count += 1

        # Obtenemos el puntaje de la última observación
        info = self.locals.get("infos", [])[0]  # Información de la última observación
        self.current_score = info.get('score', 0)  # Puntaje actual
        lives = info.get('lives', 0)  # Vidas restantes
        done = info.get('done', False)  # Si el episodio terminó

        # Actualizar el puntaje máximo
        if self.current_score > self.max_score:
            self.max_score = self.current_score

        # Registrar puntaje y tiempo
        elapsed_time_minutes = (time.time() - self.start_time) / 60
        self.times_in_minutes.append(elapsed_time_minutes)
        self.scores.append(self.current_score)

        # Imprimir logs cada log_interval pasos
        if self.step_count % self.log_interval == 0:
            elapsed_time = timedelta(seconds=int(time.time() - self.start_time))
            print(f"Tiempo transcurrido de entrenamiento: {elapsed_time} - "
                  f"Puntaje actual: {self.current_score} - "
                  f"Puntaje Máximo: {self.max_score}")

        return True  # Retornar True para continuar el entrenamiento

def main():
    # Registrar el tiempo de inicio
    start_time = time.time()
    start_datetime = datetime.now()  # Hora de inicio en formato legible

    print(f"Inicio del entrenamiento: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")

    env_num = 6
    # Crear env_num ambientes paralelos usando SubprocVecEnv
    env = SubprocVecEnv([lambda idx=i: make_env() for i in range(env_num)])  # Crear env_num entornos

    # Definir el modelo PPO
    model = PPO(
        policy="CnnPolicy",  # Usamos una política basada en convoluciones para imágenes
        env=env,
        learning_rate=2.5e-4,
        n_steps=128,
        batch_size=32,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.01,
        verbose=1,
    )

    # Crear un callback para cada ambiente
    eval_callbacks = [EvalCallback(start_time=start_time, verbose=1, log_interval=5000) for _ in range(env_num)]

    # Entrenar el modelo con el callback
    model.learn(total_timesteps=1900800, log_interval=1, callback=eval_callbacks)

    # Guardar el modelo entrenado
    model.save("ppo_gradius_model")
    print("Modelo entrenado y guardado como 'ppo_gradius_model'.")

    # Unificar los datos de todos los callbacks
    unified_times = []
    unified_scores = []
    for callback in eval_callbacks:
        unified_times.extend(callback.times_in_minutes)
        unified_scores.extend(callback.scores)

    # Guardar los datos en un archivo CSV
    save_training_data(unified_times, unified_scores, filename="training_data.csv")

    # Ordenar los datos por tiempo para un gráfico continuo
    combined_data = sorted(zip(unified_times, unified_scores))
    sorted_times, sorted_scores = zip(*combined_data)

    # Graficar puntajes vs tiempo de entrenamiento (en minutos)
    plt.figure(figsize=(10, 6))
    plt.plot(sorted_times, sorted_scores, label="Score", color="blue")

    plt.xlabel('Tiempo de entrenamiento (minutos)')
    plt.ylabel('Puntaje')
    plt.title('Puntaje vs Tiempo de Entrenamiento')
    plt.grid(True)
    plt.legend()
    plt.savefig('score_vs_time_minutes.png')  # Guardamos la gráfica
    print("Gráfico guardado como 'score_vs_time_minutes.png'.")

    # Calcular el tiempo total de ejecución
    total_time_minutes = (time.time() - start_time) / 60
    end_datetime = datetime.now()  # Hora de fin en formato legible

    # Guardar logs en un archivo
    log_filename = "ppoTrainLog.txt"
    with open(log_filename, "w") as log_file:
        log_file.write(f"La hora de inicio del entrenamiento fue: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"Fin del entrenamiento: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"El entrenamiento tomó {total_time_minutes:.2f} minutos en total.\n")

    # También imprimir en consola
    print(f"La hora de inicio del entrenamiento fue: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Fin del entrenamiento: {end_datetime.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"El entrenamiento tomó {total_time_minutes:.2f} minutos en total.")

    env.close()

if __name__ == "__main__":
    main()
