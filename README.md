# ⚗️ Otimização Não-Linear: Método de Gauss-Newton aplicado à Corrosão

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Math](https://img.shields.io/badge/Math-Optimization-orange)
![Status](https://img.shields.io/badge/Status-Research%20Complete-green)

> *Resolução de Problemas de Mínimos Quadrados Não Lineares para o Combate à Corrosão.*

---

## 📄 Sobre o Projeto

Este projeto desenvolveu um ambiente computacional para identificar parâmetros eletroquímicos fundamentais em processos de corrosão galvânica. Utilizando o **Método de Gauss-Newton** implementado manualmente em Python, o algoritmo ajusta curvas de polarização não-lineares a dados experimentais, minimizando o erro entre o modelo teórico e a realidade física.

O diferencial deste código é a implementação da **Condição de Armijo** para controle do passo (Line Search), garantindo a estabilidade numérica e a convergência global do método.

---

## 📐 Modelagem Matemática

O problema consiste em encontrar os parâmetros ótimos $\theta = [\beta_a, \beta_c, i_c, E_c]$ que minimizam a soma dos quadrados dos resíduos:

$$\min_{\theta} \sum_{k=1}^{n} (i_{exp}^{(k)} - i_{modelo}(E^{(k)}, \theta))^2$$

### A Equação Governança (Butler-Volmer)
A função não-linear que descreve a densidade de corrente ($i$) em função do potencial ($E$) é dada por:

$$i = i_{c} \left[ e^{\frac{2.303(E-E_{c})}{\beta_{a}}} - e^{\frac{2.303(E_{c}-E)}{\beta_{c}}} \right]$$

Onde os parâmetros a serem descobertos pelo algoritmo são:
* $\beta_a$: Inclinação de Tafel Anódica.
* $\beta_c$: Inclinação de Tafel Catódica.
* $i_c$: Densidade de corrente de corrosão.
* $E_c$: Potencial de corrosão.

---

## 🛠️ O Algoritmo (Implementação)

Ao invés de utilizar solvers de caixa preta, o método foi implementado "from scratch" seguindo a lógica iterativa:

1.  **Cálculo da Jacobiana ($J$):** Derivação analítica das sensibilidades de cada parâmetro.
2.  **Sistema Normal:** Resolução de $(J^T J) \Delta \theta = -J^T r$ para encontrar a direção de descida.
3.  **Step Control (Armijo):** Ajuste do tamanho do passo $\alpha$ para garantir que a função objetivo decresça a cada iteração ($f(x + \alpha d) < f(x)$).

### Snippet da Matriz Jacobiana
```python
# Trecho do código onde as derivadas parciais são calculadas manualmente
J[k, 0] = i_c * term_a * 2.303 * (E[k] - E_c) * (-1 / (beta_a**2)) # d/d(beta_a)
J[k, 1] = i_c * (-term_c) * 2.303 * (E_c - E[k]) * (-1 / (beta_c**2)) # d/d(beta_c)
