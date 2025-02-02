# Human-Mobility-Prediction-Thesis

Este repositorio contiene el código y los experimentos realizados para mi tesis sobre **Predicción de Movilidad Humana**. En la misma se aborda el desafío de la predicción de la movilidad humana mediante el uso de redes neuronales. Se implementaron diferentes arquitecturas, incluyendo una red simple para la predicción de días futuros, una red para estimar patrones semanales y una red GRU para capturar las relaciones temporales complejas en los datos. Además, se exploró el ajuste de modelos preentrenados basados en *Transformer-BERT* para evaluar su aplicabilidad en la modelización de patrones espacio-temporales.  

El enfoque seguido se basa en la segmentación temporal de los datos en ciclos semanales y horarios, con predicciones realizadas a nivel individual. La investigación busca explorar diferentes vías para el modelado de estos problemas, sentando las bases para futuras exploraciones con arquitecturas híbridas que combinen redes recurrentes con modelos basados en atención.

- **Modelos base:** Se implementa un modelo *baseline* que asume que los patrones de movilidad se repiten semanalmente.
- **Redes Neuronales Simples (Densas):** Se utilizan para establecer un punto de comparación inicial.
- **Modelos GRU:** Se experimenta con diferentes configuraciones de entrada para evaluar su impacto en la predicción.
- **Transformers (*Fine-tuning* de BERT):** Se adapta BERT al problema de movilidad para capturar dependencias a largo plazo.

Los modelos se evaluaron con métricas como **MAE, LPP, DTW y GeoBLEU**, y los experimentos fueron ejecutados en servidores de alto rendimiento utilizando **Google Colab**.

## Estructura del repositorio

### 📂 Notebooks

Los siguientes notebooks contienen la implementación de los modelos y experimentos:

- [📘 Preprocesamiento de Datos](./notebooks/initial_analisys.ipynb) - Normalización, manejo de valores faltantes y generación de secuencias.
- [📘 Modelo Baseline](./notebooks/NaiveWeekRepeatPredictor.ipynb) - Implementación del modelo base basado en recurrencia semanal.
- [📘 Redes Neuronales Simples - Predicción de un día](./notebooks/LastWeekSimpleNN.ipynb) - Red neuronal densa para predecir la movilidad de un solo día a partir de la semana previa.
- [📘 Redes Neuronales Simples - Predicción de una semana](./notebooks/SimpleNN.ipynb) - Red neuronal densa para predecir la movilidad de una semana completa utilizando los datos de la semana anterior.
- [📘 Modelos GRU](./notebooks/GRU.ipynb) - Implementación de las variantes de GRU con diferentes características de entrada.
- [📘 Modelo Transformer basado en BERT - LPBert](./notebooks/BertLP.ipynb) - Ajuste fino de BERT para predicción de movilidad.

### 📄 Documento de la Tesis

Puedes acceder al documento completo de la tesis en formato PDF aquí:

📄 [Descargar](link_al_pdf_tesis)

---

## 🔧 Configuración del Entorno

Para ejecutar los experimentos, primero es necesario configurar el entorno virtual y las dependencias. Se han automatizado las tareas con `make`:

### Crear el entorno virtual

```bash
make venv
```

### Generar el archivo requirements.txt

```bash
make requirements
```

### Instalar dependencias desde requirements.txt

```bash
make install
```

### Limpiar el entorno virtual y requirements.txt

```bash
make clean
```
