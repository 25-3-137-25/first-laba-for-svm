import numpy as np
import pandas as pd

# Исходные данные
A = np.array([
    [12.46, 1.35, -2.14, 0.86],
    [-1.28, 10.73, 1.66, -1.45],
    [2.17, -0.95, 13.52, 1.34],
    [1.08, 2.26, -1.37, 11.29]
], dtype=float)

b = np.array([17.25, 11.38, 24.61, 15.74], dtype=float)

print("Матрица A:\n", A)
print("Вектор b:\n", b)

# Проверка числа обусловленности
cond_A = np.linalg.cond(A)
print(f"Число обусловленности cond(A): {cond_A:.4f}")

# Эталонное решение (NumPy)
x_lib = np.linalg.solve(A, b)
print("\n Эталонное решение (NumPy)")
print(f"x_lib = {x_lib}")


# Функция для расчета невязки ||Ax - b||
def calculate_residual(A, x, b):
    residual_vector = A @ x - b
    return np.linalg.norm(residual_vector)


res_lib = calculate_residual(A, x_lib, b)
print(f"Невязка эталона: {res_lib:.2e}")


# Метод Гаусса с выбором ведущего элемента
def gauss_method(A, b):
    n = len(b)
    
    A = A.copy().astype(float)
    b = b.copy().astype(float)
  
    for k in range(n):
        
        max_row = k + np.argmax(np.abs(A[k:n, k]))
        
        if max_row != k:
            A[[k, max_row]] = A[[max_row, k]]
            b[[k, max_row]] = b[[max_row, k]]
        
        for i in range(k + 1, n):
            m = A[i, k] / A[k, k]
            A[i, k:] -= m * A[k, k:]
            b[i] -= m * b[k]
    
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i + 1:], x[i + 1:])) / A[i, i]
    return x


x_gauss = gauss_method(A, b)
print("\nМетод Гаусса")
print(f"x_gauss = {x_gauss}")
print(f"Отличие от NumPy: {np.linalg.norm(x_gauss - x_lib):.2e}")
print(f"Невязка: {calculate_residual(A, x_gauss, b):.2e}")


# Метод простой итерации
def simple_iteration(A, b, x0, eps=1e-6, max_iter=1000):
    n = len(b)
    x = np.array(x0, dtype=float, copy=True)
    x_new = np.zeros(n)

    for k in range(max_iter):
        for i in range(n):
            s = sum(A[i][j] * x[j] for j in range(n) if j != i)
            x_new[i] = (b[i] - s) / A[i][i]

        diff = np.linalg.norm(x_new - x)
        x = x_new.copy()
        print(f"k={k + 1}, x={np.array2string(x, precision=10)}, ε_k={diff:.2e}")

        if diff < eps:
            return x, k + 1

    raise RuntimeError(f"Простая итерация: точность {eps:g} не достигнута за {max_iter} итераций.")


# Метод Зейделя
def zeidel_method(A, b, x0, eps=1e-6, max_iter=1000):
    n = len(b)
    x = np.array(x0, dtype=float, copy=True)

    for k in range(max_iter):
        x_old = x.copy()
        for i in range(n):
            s1 = sum(A[i][j] * x[j] for j in range(i))  # Уже новые значения
            s2 = sum(A[i][j] * x_old[j] for j in range(i + 1, n))  # Старые значения
            x[i] = (b[i] - s1 - s2) / A[i][i]

        diff = np.linalg.norm(x - x_old)
        print(f"k={k + 1}, x={np.array2string(x, precision=10)}, ε_k={diff:.2e}")
        if diff < eps:
            return x, k + 1

    raise RuntimeError(f"Зейдель: точность {eps:g} не достигнута за {max_iter} итераций.")

print("\nСравнение итерационных методов")

# 1. Метод простой итерации
print("\nМетод простой итерации:")
x_iter, iters_iter = simple_iteration(A, b, np.zeros(4), eps=1e-6)
res_iter = calculate_residual(A, x_iter, b)
err_iter = np.linalg.norm(x_iter - x_lib)

print(f"  Решение: {x_iter}")
print(f"  Итераций: {iters_iter}")
print(f"  Ошибка ||x - x_lib||: {err_iter:.2e}")
print(f"  Невязка ||r||: {res_iter:.2e}")

# 2. Метод Зейделя
print("\nМетод Зейделя:")
x_zeid, iters_zeid = zeidel_method(A, b, np.zeros(4), eps=1e-6)
res_zeid = calculate_residual(A, x_zeid, b)
err_zeid = np.linalg.norm(x_zeid - x_lib)

print(f"  Решение: {x_zeid}")
print(f"  Итераций: {iters_zeid}")
print(f"  Ошибка ||x - x_lib||: {err_zeid:.2e}")
print(f"  Невязка ||r||: {res_zeid:.2e}")

print(f"\nВывод: Метод Зейделя сошелся быстрее (итераций: {iters_zeid} vs {iters_iter})")

# Вычислительные эксперименты
methods = {"Простая итерация": simple_iteration, "Зейдель": zeidel_method}

print("\nЭксперимент 1: Разные начальные приближения (оба метода, eps=1e-6)")
initial_guesses = {
    "Нулевой вектор": np.zeros(4),
    "Единичный вектор": np.ones(4),
    "Произвольный": np.array([10.0, -5.0, 2.0, 0.5])
}

results_exp1 = []
for method_name, method in methods.items():
    for name, x0 in initial_guesses.items():
        print(f"\n{method_name}, {name}:")
        x_z, iters_z = method(A, b, x0, eps=1e-6)
        res_z = calculate_residual(A, x_z, b)
        err_z = np.linalg.norm(x_z - x_lib)
        results_exp1.append([method_name, name, x_z.copy(), iters_z, f"{err_z:.2e}", f"{res_z:.2e}"])
        print(f"{name}: Итераций={iters_z}, Ошибка={err_z:.2e}, Невязка={res_z:.2e}")

df_exp1 = pd.DataFrame(results_exp1,
                       columns=["Метод", "Начальное приближение", "Решение", "Итерации", "Ошибка ||x-x_lib||", "Невязка ||r||"])
print(df_exp1.to_string(index=False))

print("\nЭксперимент 2: Влияние точности eps (оба метода, x0=0)")
epsilons = [1e-2, 1e-4, 1e-6, 1e-8]
results_exp2 = []
for method_name, method in methods.items():
    for eps in epsilons:
        print(f"\n{method_name}, eps={eps:.0e}:")
        x_z, iters_z = method(A, b, np.zeros(4), eps=eps)
        res_z = calculate_residual(A, x_z, b)
        results_exp2.append([method_name, f"{eps:.0e}", iters_z, f"{res_z:.2e}"])
        print(f"eps={eps:.0e}: Итераций={iters_z}, Невязка={res_z:.2e}")

df_exp2 = pd.DataFrame(results_exp2, columns=["Метод", "Точность (eps)", "Итерации", "Невязка ||r||"])
print(df_exp2.to_string(index=False))

print("\nЭксперимент 3: Округление коэффициентов")
# Два знака сохраняют данные, один и ноль знаков изменяют их.
for digits in [2, 1, 0]:
    A_rounded = np.round(A, digits)
    b_rounded = np.round(b, digits)

    x_lib_rounded = np.linalg.solve(A_rounded, b_rounded)
    delta_x = np.linalg.norm(x_lib - x_lib_rounded)

    print(f"\nОкругление до {digits} знаков. Решение: {x_lib_rounded}")
    print(f"Изменение решения ||x_orig - x_rounded||: {delta_x:.2e}")
    print(f"Число обусловленности округленной матрицы: {np.linalg.cond(A_rounded):.4f}")
