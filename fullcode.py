import streamlit as st
import sympy as sp
import numpy as np

import re


def parse_latex(latex_str):
    """Parses LaTeX matrix expressions and identifies operations"""
    latex_str = latex_str.strip()

    # Identify if a specific operation is required
    operation = None
    match = re.match(r'^(Tr|Inv|Det|Trx|E|Ev|Dim|R)\s*=\s*(.*)', latex_str, re.IGNORECASE)
    if match:
        operation = match.group(1).lower()
        latex_str = match.group(2).strip()

    # Extract matrices
    matrix_strings = re.findall(r'\\begin{bmatrix}(.+?)\\end{bmatrix}', latex_str, re.DOTALL)

    matrices = []
    for mat_str in matrix_strings:
        rows = mat_str.strip().split(r'\\')
        matrix = [list(map(int, row.split('&'))) for row in rows]
        matrices.append(sp.Matrix(matrix))

    # Extract operations (+, -, *)
    operations = re.findall(r'(\+|\-|\*)', latex_str)

    return operation, matrices, operations


def compute_result(operation, matrices, operations):
    """Computes the result based on the operation"""
    if not matrices:
        return "No valid matrix detected!"

    # Handle special operations (Transpose, Inverse, Determinant, etc.)
    if operation == "tr":  # Transpose
        return matrices[0].T
    elif operation == "inv":  # Inverse
        if matrices[0].det() == 0:
            return "Matrix is singular, inverse does not exist!"
        return matrices[0].inv()
    elif operation == "det":  # Determinant
        return matrices[0].det()
    elif operation == "trx":  # Trace
        return matrices[0].trace()
    elif operation == "e":  # Eigenvalues
        return matrices[0].eigenvals()
    elif operation == "ev":  # Eigenvectors
        return matrices[0].eigenvects()
    elif operation == "dim":  # Dimension (Rank of matrix)
        return matrices[0].rank()
    elif operation == "r":  # Rank
        return matrices[0].rank()

    # Perform matrix operations if multiple matrices are present
    if len(matrices) > 1 and operations:
        result = matrices[0]  # Start with the first matrix
        for i in range(1, len(matrices)):
            if operations[i - 1] == "+":
                if result.shape == matrices[i].shape:  # Check dimension match
                    result += matrices[i]
                else:
                    return "Addition not possible due to dimension mismatch!"
            elif operations[i - 1] == "-":
                if result.shape == matrices[i].shape:  # Check dimension match
                    result -= matrices[i]
                else:
                    return "Subtraction not possible due to dimension mismatch!"
            elif operations[i - 1] == "*":
                if result.shape[1] == matrices[i].shape[0]:  # Check matrix multiplication rule
                    result *= matrices[i]
                else:
                    return "Multiplication not possible due to shape mismatch!"
        return result

    return "Invalid operation!"


def compute_result(operation, matrices):
    """Computes the result based on the selected operation."""
    if not matrices:
        return "No valid matrix detected!", ""

    result = None
    latex_expression = ""

    if operation == "Transpose":
        result = matrices[0].T
        latex_expression = f"Tr = {generate_latex(matrices[0])}^T"
    elif operation == "Determinant":
        result = matrices[0].det()
        latex_expression = f"Det = \\det({generate_latex(matrices[0])})"
    elif operation == "Trace":
        result = matrices[0].trace()
        latex_expression = f"Trx = \\text{{trace}}({generate_latex(matrices[0])})"
    elif operation == "Rank":
        result = matrices[0].rank()
        latex_expression = f"R = \\text{{rank}}({generate_latex(matrices[0])})"
    elif operation == "Dimension":
        result = matrices[0].rank()
        latex_expression = f"Dim = \\text{{dim}}({generate_latex(matrices[0])})"
    elif operation == "Inverse":
        result = matrices[0].inv() if matrices[0].det() != 0 else "Matrix is singular, inverse does not exist!"
        latex_expression = f"Inv = {generate_latex(matrices[0])}^{-1}"
    elif operation == "Eigenvalues":
        result = matrices[0].eigenvals()
        latex_expression = f"E = \\text{{Eigenvalues of }} {generate_latex(matrices[0])}"
    elif operation == "Eigenvectors":
        result = matrices[0].eigenvects()
        latex_expression = f"Ev = \\text{{Eigenvectors of }} {generate_latex(matrices[0])}"
    elif operation in ["Addition", "Subtraction", "Multiplication"]:
        result = matrices[0]
        latex_expression = generate_latex(matrices[0])
        for i in range(1, len(matrices)):
            if operation == "Addition":
                result += matrices[i]
                latex_expression += f" + {generate_latex(matrices[i])}"
            elif operation == "Subtraction":
                result -= matrices[i]
                latex_expression += f" - {generate_latex(matrices[i])}"
            elif operation == "Multiplication":
                if result.shape[1] == matrices[i].shape[0]:
                    result *= matrices[i]
                    latex_expression += f" \\times {generate_latex(matrices[i])}"
                else:
                    return "Multiplication not possible due to shape mismatch!", ""

    return result, latex_expression


# Streamlit UI
st.title("Matrix Calculator")

# Buttons to choose input mode
calc_mode = st.radio("Choose an input method:", ["LaTeX Code", "Matrix Calculator"])

if calc_mode == "LaTeX Code":
    st.markdown("[Click here to open LaTeX editor](https://webdemo.myscript.com/views/math/index.html#)",
                unsafe_allow_html=True)
    latex_input = st.text_area("Enter LaTeX matrix expression:", "")
    if st.button("Compute LaTeX Expression"):
        if latex_input:
            operation, matrices, operations = parse_latex(latex_input)
            result = compute_result(operation, matrices, operations)
            st.write("### Result:")
            st.latex(sp.latex(result))
        else:
            st.error("Please enter a valid LaTeX matrix expression!")

elif calc_mode == "Matrix Calculator":
    st.write("Matrix Calculator")

    # Available Operations
    operation = st.selectbox("Choose an operation:", [
        "Addition",
        "Subtraction",
        "Multiplication",
        "Transpose",
        "Inverse",
        "Determinant",
        "Rank",
        "Trace",
        "Eigenvalues",
        "Eigenvectors"
    ])


    # Function to get matrix input
    def get_matrix_input(rows, cols, name):
        matrix = []
        for i in range(rows):
            row = st.text_input(f"{name} - Row {i + 1} (space-separated values)")
            matrix.append(list(map(float, row.split()))) if row else matrix.append([0] * cols)
        return np.array(matrix)


    # Input Matrices Based on Operation
    if operation in ["Addition", "Subtraction", "Multiplication"]:
        n = st.number_input("Enter rows for Matrix A (n):", min_value=1, step=1, value=2)
        m = st.number_input("Enter columns for Matrix A / rows for Matrix B (m):", min_value=1, step=1, value=2)
        p = m if operation != "Multiplication" else st.number_input("Enter columns for Matrix B (p):", min_value=1,
                                                                    step=1, value=2)

        st.header("Enter Matrix A Elements")
        matrix_a = get_matrix_input(n, m, "Matrix A")

        st.header("Enter Matrix B Elements")
        matrix_b = get_matrix_input(m, p, "Matrix B")

        if operation in ["Addition", "Subtraction"] and matrix_a.shape != matrix_b.shape:
            st.error("Matrix dimensions must match for addition or subtraction!")
        elif operation == "Multiplication" and matrix_a.shape[1] != matrix_b.shape[0]:
            st.error("Matrix A's columns must match Matrix B's rows for multiplication!")
    else:
        n = st.number_input("Enter the dimension (n x n) of the matrix:", min_value=1, step=1, value=2)
        st.header("Enter Matrix Elements")
        matrix_a = get_matrix_input(n, n, "Matrix A")

    st.write("Input Matrix A:")
    st.write(matrix_a)
    if operation in ["Addition", "Subtraction", "Multiplication"]:
        st.write("Input Matrix B:")
        st.write(matrix_b)

    # Perform Selected Operation
    if st.button("Compute"):
        try:
            if operation == "Addition":
                result = matrix_a + matrix_b
            elif operation == "Subtraction":
                result = matrix_a - matrix_b
            elif operation == "Multiplication":
                result = np.dot(matrix_a, matrix_b)
            elif operation == "Transpose":
                result = np.transpose(matrix_a)
            elif operation == "Inverse":
                result = np.linalg.inv(matrix_a)
            elif operation == "Determinant":
                result = np.linalg.det(matrix_a)
            elif operation == "Rank":
                result = np.linalg.matrix_rank(matrix_a)
            elif operation == "Trace":
                result = np.trace(matrix_a)
            elif operation == "Eigenvalues":
                result = np.linalg.eigvals(matrix_a)
            elif operation == "Eigenvectors":
                _, eigenvectors = np.linalg.eig(matrix_a)
                result = eigenvectors.tolist()
            else:
                result = "Invalid operation selected."

            st.write("Result:")
            st.write(result)
        except Exception as e:
            st.error(f"Error: {e}")