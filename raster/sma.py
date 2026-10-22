"""
Spectral Mixture Analysis
"""



import numpy as np
from scipy.optimize import nnls

# 1. Definir os 4 Endmembers Globais (Azul, Verde, Vermelho, NIR, SWIR1, SWIR2)
# Valores hipotéticos de refletância (L2A) multiplicados por 10000
endmembers = np.array([
    [ 200,  400,  300, 3500, 1500,  600], # PV (Floresta Verde)
    [ 800, 1200, 1500, 2200, 3500, 2500], # NPV (Madeira Seca/Palha)
    [1200, 1800, 2500, 3100, 4500, 3800], # Solo Exposto (Saprólito)
    [ 100,  150,  100,  100,   50,   20]  # Sombra / Água Profunda
])

# Transpor para matriz M (Bandas x Endmembers): formato (6, 4)
M = endmembers.T 

# 2. Preparar os dados do Sentinel-2
# sentinel_stack possui shape (6, altura, largura)
c, h, w = sentinel_stack.shape

# Achatar a imagem para (6, pixels_totais)
pixels = sentinel_stack.reshape(c, h * w)

# 3. Inicializar matrizes de saída para as frações e o erro (RMSE)
fracoes = np.zeros((4, h * w), dtype=np.float32)
rmse = np.zeros((h * w), dtype=np.float32)

# 4. Calcular o NNLS pixel a pixel
# Como o NNLS do SciPy não é totalmente vetorizado por padrão, iteramos sobre os pixels.
# Para datasets massivos, bibliotecas como pysptools ou algoritmos otimizados em Numba/Cython aceleram este laço.
for i in range(h * w):
    pixel_espectro = pixels[:, i]
    
    # Executa Mínimos Quadrados Não-Negativos
    # Retorna as frações (f) e o resíduo da equação
    f, residuo = nnls(M, pixel_espectro)
    
    # Opcional (FCLS): Normalizar para que a soma das frações seja exatamente 1 (100%)
    soma = np.sum(f)
    if soma > 0:
        f = f / soma
        
    fracoes[:, i] = f
    rmse[i] = residuo

# 5. Redimensionar de volta para a geometria espacial da imagem
fracoes_img = fracoes.reshape(4, h, w)
rmse_img = rmse.reshape(h, w)

# Extração final das camadas individuais para empilhar no tensor
pv_img     = fracoes_img[0, :, :]
npv_img    = fracoes_img[1, :, :]
solo_img   = fracoes_img[2, :, :]
sombra_img = fracoes_img[3, :, :]