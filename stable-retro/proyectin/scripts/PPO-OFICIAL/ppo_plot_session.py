import pandas as pd
import matplotlib.pyplot as plt

# Cargar los datos de ambos entrenamientos
df_train = pd.read_csv("training_data.csv", names=["Time (minutes)", "Score"])
df_continue = pd.read_csv("training_data_continue.csv", names=["Time (minutes)", "Score"])

# Graficar ambos entrenamientos
plt.plot(df_train["Time (minutes)"], df_train["Score"], label="Entrenamiento Inicial", color="blue")
plt.plot(df_continue["Time (minutes)"], df_continue["Score"], label="Entrenamiento Continuado", color="red")

# Personalizar la gráfica
plt.xlabel("Tiempo (minutos)")
plt.ylabel("Puntaje")
plt.legend()
plt.title("Entrenamiento PPO de GradiusIII-Snes")
plt.grid(True)

# Guardar la gráfica
plt.savefig("training_comparison.png")
plt.show()
