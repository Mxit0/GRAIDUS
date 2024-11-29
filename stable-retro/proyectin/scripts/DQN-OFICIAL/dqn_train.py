import retro
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import time
from datetime import datetime, timedelta
from stable_baselines3 import DQN
from stable_baselines3.common.atari_wrappers import WarpFrame, ClipRewardEnv
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import os
import csv

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
    def __init__(self, start_time, verbose=0, log_interval=1000, csv_filename="training_data.csv"):
        super(EvalCallback, self).__init__(verbose)
        self.max_score = -np.inf  # Puntaje máximo inicial
        self.current_score = 0  # Puntaje actual
        self.log_interval = log_interval  # Intervalo para imprimir logs
        self.start_time = start_time  # Tiempo de inicio del entrenamiento
        self.step_count = 0  # Contador de pasos
        self.times_in_minutes = []  # Almacenar tiempos en minutos
        self.scores = []  # Almacenar puntajes
        self.csv_filename = csv_filename  # Nombre del archivo CSV donde se guardarán los datos

    def _on_step(self) -> bool:
        """
        Este método se llama después de cada paso de entrenamiento.
        Guarda el puntaje máximo y actual en cada paso, y registra el tiempo en minutos.
        """
        self.step_count += 1

        # Obtenemos el puntaje de la última observación
        info = self.locals.get("infos", [])[0]  # Información de la última observación
        self.current_score = info.get('score', 0)  # Puntaje actual

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

        # Guardar datos en el archivo CSV
        self.save_to_csv()

        return True  # Retornar True para continuar el entrenamiento

    def save_to_csv(self):
        """Guardar los datos de tiempos y puntajes en un archivo CSV."""
        with open(self.csv_filename, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([self.times_in_minutes[-1], self.scores[-1]])

def main():
    # Registrar el tiempo de inicio
    start_time = time.time()
    start_datetime = datetime.now()  # Hora de inicio en formato legible

    print(f"Inicio del entrenamiento: {start_datetime.strftime('%Y-%m-%d %H:%M:%S')}")

    env_num = 6
    # Crear env_num ambientes paralelos usando SubprocVecEnv
    env = SubprocVecEnv([lambda idx=i: make_env() for i in range(env_num)])  # Crear env_num entornos

    # Definir el modelo DQN
    model = DQN(
        policy="CnnPolicy",  # Usamos una política basada en convoluciones para imágenes
        env=env,
        learning_rate=2.5e-4,
        buffer_size=50000,
        batch_size=32,
        n_warmup_steps=1000,
        gamma=0.99,
        target_network_update_freq=100,
        train_freq=4,
        verbose=1,
    )

    # Crear un callback para recolectar datos durante el entrenamiento
    eval_callback = EvalCallback(start_time=start_time, verbose=1, log_interval=5000)

    # Entrenar el modelo con el callback
    model.learn(total_timesteps=1900800, log_interval=1, callback=eval_callback)

    # Guardar el modelo entrenado
    model.save("dqn_gradius_model")
    print("Modelo entrenado y guardado como 'dqn_gradius_model'.")

    # Unificar los datos de todos los callbacks
    unified_times = []
    unified_scores = []
    unified_times.extend(eval_callback.times_in_minutes)
    unified_scores.extend(eval_callback.scores)

    # Guardar los datos en un archivo CSV
    save_training_data(unified_times, unified_scores, filename="training_data.csv")

    env.close()

if __name__ == "__main__":
    main()
