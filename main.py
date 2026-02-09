import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# =============================================================================
# 1. DEFINIÇÃO DOS DADOS E PARÂMETROS
# =============================================================================

# Dados experimentais extraídos do PDF [cite: 189, 190]
E_exp = np.array([-50, -45, -40, -35, -30, -25, -20, -15, -10])
i_exp = np.array([-1.25252, -1.01743, -0.74747, -0.41947, 0.00000, 
                   0.55940, 1.32936, 2.41308, 3.96163])

# Configurações do Algoritmo [cite: 256-261]
TOLERANCIA = 1e-7
MAX_ITER = 500
ALPHA_INICIAL = 1.0  # Passo inicial para Armijo
C1 = 1e-4            # Constante de Armijo
THETA = 0.5          # Fator de redução do passo

# Chutes iniciais (Beta_a, Beta_c, i_c, E_c) [cite: 280-283]
# Valores aleatórios ou fixos conforme testes do PDF
params = np.array([10.0, 10.0, 10.0, 10.0]) 

# =============================================================================
# 2. FUNÇÕES DO MODELO MATEMÁTICO
# =============================================================================

def polarizacao_model(E, p):
    """
    Função não-linear de polarização (Butler-Volmer simplificado).
    p[0]: beta_a (Inclinação Anódica)
    p[1]: beta_c (Inclinação Catódica)
    p[2]: i_c    (Corrente de Corrosão)
    p[3]: E_c    (Potencial de Corrosão)
    
    Fórmula: i = ic * (exp(...) - exp(...)) [cite: 222]
    """
    beta_a, beta_c, i_c, E_c = p
    
    term_a = np.exp((2.303 * (E - E_c)) / beta_a)
    term_c = np.exp((2.303 * (E_c - E)) / beta_c)
    
    return i_c * (term_a - term_c)

def calcular_jacobiana(E, p):
    """
    Calcula a Matriz Jacobiana com as derivadas parciais.
    Baseado nas fórmulas matemáticas do PDF [cite: 308-314].
    """
    beta_a, beta_c, i_c, E_c = p
    m = len(E)
    J = np.zeros((m, 4))
    
    # Termos comuns para evitar repetição
    term_a = np.exp((2.303 * (E - E_c)) / beta_a)
    term_c = np.exp((2.303 * (E_c - E)) / beta_c)
    
    # Derivada parcial d/d(beta_a)
    # J[:, 0] = i_c * term_a * 2.303 * (E - E_c) * (-1 / beta_a**2)
    J[:, 0] = i_c * term_a * (-2.303 * (E - E_c) / (beta_a**2))
    
    # Derivada parcial d/d(beta_c)
    # J[:, 1] = i_c * (-term_c) * 2.303 * (E_c - E) * (-1 / beta_c**2)
    # Simplificando a regra da cadeia:
    J[:, 1] = -i_c * term_c * (-2.303 * (E_c - E) / (beta_c**2))
    
    # Derivada parcial d/d(i_c)
    J[:, 2] = term_a - term_c
    
    # Derivada parcial d/d(E_c)
    # Regra da cadeia para ambos os termos exponenciais
    deriv_exp_a = term_a * (-2.303 / beta_a)
    deriv_exp_c = -term_c * (2.303 / beta_c) # O menos vem da fórmula i_c*(A - C)
    J[:, 3] = i_c * (deriv_exp_a - deriv_exp_c)
    
    return J

def funcao_objetivo(p):
    """Calcula a soma dos quadrados dos resíduos (SSE) e o vetor de resíduos."""
    i_est = polarizacao_model(E_exp, p)
    res = i_est - i_exp
    sse = 0.5 * np.sum(res**2) # [cite: 303]
    return sse, res

# =============================================================================
# 3. ALGORITMO DE GAUSS-NEWTON COM ARMIJO
# =============================================================================

print(f"--- INICIANDO OTIMIZAÇÃO: {datetime.now()} ---")
print(f"Chute Inicial: {params}\n")

contador = 0
sse_atual, res = funcao_objetivo(params)
historico_sse = [sse_atual]

while sse_atual > TOLERANCIA and contador < MAX_ITER:
    
    # 1. Calcular Jacobiana e Gradiente
    J = calcular_jacobiana(E_exp, params)
    gradiente = J.T @ res  # Gradiente = J^T * r [cite: 328]
    hessiana_aprox = J.T @ J # Hessiana = J^T * J [cite: 325]
    
    # 2. Calcular direção de descida (pk)
    # Resolve o sistema linear (H * pk = -gradiente) ou usa pseudo-inversa
    # pk = - np.linalg.pinv(hessiana_aprox) @ gradiente [cite: 326, 330]
    pk = -np.linalg.pinv(hessiana_aprox) @ gradiente
    
    # 3. Condição de Armijo (Line Search) [cite: 157-159]
    alpha = ALPHA_INICIAL
    params_temp = params + alpha * pk
    sse_temp, _ = funcao_objetivo(params_temp)
    
    # Verifica se o passo satisfaz Armijo: f(x + alpha*p) <= f(x) + c1*alpha*grad*p
    while sse_temp > sse_atual + C1 * alpha * np.dot(gradiente, pk):
        alpha *= THETA  # Reduz o passo
        params_temp = params + alpha * pk
        sse_temp, _ = funcao_objetivo(params_temp)
        
        # Proteção contra passo muito pequeno
        if alpha < 1e-8:
            break
            
    # 4. Atualizar Parâmetros
    params = params_temp
    sse_atual, res = funcao_objetivo(params)
    historico_sse.append(sse_atual)
    contador += 1
    
    # Log a cada 5 iterações
    if contador % 5 == 0:
        print(f"Iteração {contador}: SSE = {sse_atual:.6f}")

# =============================================================================
# 4. RESULTADOS E VISUALIZAÇÃO
# =============================================================================

print("\n" + "="*40)
print("RESULTADO FINAL")
print("="*40)
print(f"Iterações: {contador}")
print(f"Erro Final (SSE): {sse_atual:.8f}")

# Criação do DataFrame para exibição bonita [cite: 442-448]
df_resultado = pd.DataFrame({
    'Parâmetro': ['Beta_a', 'Beta_c', 'i_c', 'E_c'],
    'Valor Otimizado': params
})
print(df_resultado)

# Geração do gráfico [cite: 497-504]
E_plot = np.linspace(min(E_exp), max(E_exp), 100)
i_plot = polarizacao_model(E_plot, params)

plt.figure(figsize=(8, 5))
plt.style.use('ggplot') # Estilo usado no PDF
plt.scatter(E_exp, i_exp, color='red', label='Dados Experimentais', zorder=5)
plt.plot(E_plot, i_plot, color='black', linewidth=2, label='Ajuste Gauss-Newton')

plt.title(f'Ajuste de Curva - Convergência em {contador} iterações')
plt.xlabel('Potencial E (mV)')
plt.ylabel('Corrente i (A)')
plt.legend()
plt.grid(True)

# Salvar gráfico para usar no README se quiser
plt.savefig('resultado_ajuste.png')
plt.show()
