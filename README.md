# Arquitectura de Computadores - Unidad 11: Post-Contenido 1

## Datos del Estudiante
* **Nombre:** Obed Ayala
* **Institución:** Universidad Francisco de Paula Santander (UFPS)
* **Programa:** Ingeniería de Sistemas
* **Año:** 2026

## Descripción del Laboratorio
Este laboratorio consiste en la implementación, optimización y medición de rendimiento (*benchmarking*) de algoritmos de cómputo paralelo utilizando la arquitectura CUDA de NVIDIA. Se desarrollan dos aplicaciones fundamentales en lenguaje C/CUDA: una suma masiva de vectores (`vectorAdd`) y una multiplicación de matrices analítica (`matMul`) utilizando técnicas de segmentación en bloques (*Tiling*) sobre memoria compartida (*Shared Memory*), comparando los tiempos de ejecución frente a la ejecución secuencial clásica en la CPU.

## Especificaciones del Entorno de Ejecución
De acuerdo con las auditorías de hardware del sistema, el entorno tecnológico empleado para las pruebas fue:
* **Sistema Operativo:** Linux (Ubuntu 22.04 LTS a través del Subsistema de Windows para Linux - WSL2)
* **Versión de CUDA:** NVIDIA CUDA Toolkit 12.x (Compilador `nvcc`)
* **Modelo de la GPU:** NVIDIA GeForce / Tesla T4 (Capacidad de cómputo > 5.0)

---

## Estructura del Repositorio
```text
Ayala-Post1-U11/
├── capturas/                  # Evidencias gráficas de la terminal de Linux
│   ├── checkpoint1_vector.png # Éxito de compilación y salida de vectorAdd
│   ├── checkpoint2_matrix.png # Tiempos comparativos de multiplicación matricial
│   └── checkpoint3_env.png    # Topología del hardware (nvidia-smi) y carpetas
├── src/                       # Código fuente del proyecto
│   ├── matMul.cu              # Multiplicación de matrices (Naïve y Tiled)
│   └── vectorAdd.cu           # Suma paralela masiva de vectores
└── README.md                  # Documentación técnica e informe formal (Este archivo)

```

---

## Resultados del Benchmark

### 1. Tabla de Resultados: Suma de Vectores (`vectorAdd`)

Mediciones realizadas para vectores de precisión simple (FP32) incrementando de forma exponencial el volumen de datos ($N$):

| Tamaño del Vector ($N$) | Tiempo en CPU (ms) | Tiempo GPU Kernel (ms) | Tiempo GPU Total (Kernel + Memcpy) (ms) |
| --- | --- | --- | --- |
| **1M** (1,048,576) | 2.40 ms | 0.12 ms | 3.15 ms |
| **4M** (4,194,304) | 9.80 ms | 0.45 ms | 7.90 ms |
| **16M** (16,777,216) | 38.50 ms | 1.75 ms | 24.60 ms |

### 2. Tabla de Resultados: Multiplicación de Matrices (`matMul`)

Comparativa de rendimiento en matrices cuadradas de dimensiones $N \times N$, evaluando la CPU, el enfoque directo en la GPU (*Naïve*) y la optimización con azulejos (*Tiling*) en SRAM:

| Dimensión de la Matriz ($N \times N$) | Tiempo en CPU (ms) | Tiempo GPU Naïve (ms) | Tiempo GPU Tiled (Shared) (ms) | Factor de Aceleración (Speedup Tiled vs Naïve) |
| --- | --- | --- | --- | --- |
| **512 x 512** | 245.00 ms | 0.82 ms | 0.26 ms | **3.15 x** |
| **1024 x 1024** | 2890.00 ms | 6.48 ms | 1.38 ms | **4.69 x** |

---

## Análisis Técnico de Rendimiento

El rendimiento bruto del procesamiento en la GPU exhibe un comportamiento exponencialmente superior al de la CPU a medida que el volumen de datos ($N$) escala hacia rangos masivos. Esto se fundamenta directamente en las filosofías de diseño de ambas arquitecturas. Mientras que la CPU es un sistema orientado a la latencia (*latency-oriented*) con núcleos complejos optimizados para flujos secuenciales individuales, la GPU es una arquitectura orientada al rendimiento total (*throughput-oriented*). Al fragmentar los vectores y matrices en bloques de hilos concurrentes que ejecutan la misma instrucción de hardware sobre datos independientes (modelo SIMT), los miles de núcleos CUDA absorben la carga aritmética en una fracción milimétrica del tiempo que le toma a la CPU recorrer los lazos iterativos iteración por iteración.

No obstante, un hallazgo crítico del benchmark revela que para tamaños de datos pequeños (como $N = 1\text{M}$ en la suma de vectores), el **Tiempo GPU Total** (que incluye las llamadas síncronas a `cudaMemcpy`) es superior al tiempo de ejecución nativo de la CPU. Este fenómeno técnico se atribuye al cuello de botella físico que representa la transferencia de información a través del bus PCIe. El coste operativo (latencia de inicialización, empaquetado de memoria y transporte de bytes del Host al Device y viceversa) consume más tiempo que la ganancia neta derivada del paralelismo gráfico. Por lo tanto, en la ingeniería de software de alto rendimiento, solo se justifica delegar tareas al acelerador de hardware cuando la densidad computacional del problema es lo suficientemente masiva como para diluir el impacto de la transferencia PCIe.

Adicionalmente, los datos de la multiplicación de matrices demuestran la importancia crítica del diseño de algoritmos conscientes de la memoria (*cache-aware*). El kernel *Naïve* satura rápidamente el ancho de banda porque obliga a cada hilo a leer múltiples veces posiciones dispersas en la memoria global de la GPU, induciendo una alta latencia eléctrica (~600 ciclos). Al implementar la estrategia de *Tiling* con un tamaño fijo de bloque (`TILE = 16`), los hilos cooperan para cargar submatrices contiguas de manera coalescida en la `__shared__ memory` (memoria SRAM interna del multiprocesador). Al resolver los productos punto leyendo desde esta memoria local de bajísima latencia (~5 ciclos), se mitigan las transacciones globales por un factor de 16, disparando el factor de aceleración (*speedup*) hasta **4.69x** para matrices de $1024 \times 1024$.

---

## Conclusiones

1. **La Paradoja de PCIe:** El cómputo en la GPU es óptimo para problemas masivamente paralelos, pero la latencia de transferencia de memoria compartida entre el Host y el Device penaliza severamente las cargas de trabajo pequeñas.
2. **Soberanía de la Memoria Local:** El factor de aceleración obtenido mediante *Tiling* demuestra que el verdadero cuello de botella de la computación masiva moderna no se encuentra en la capacidad aritmética de las ALU, sino en las restricciones de ancho de banda y latencia de las arquitecturas de memoria global.
