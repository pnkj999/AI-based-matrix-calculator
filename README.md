# matrix-calculator
We have created a tool for basic  real matrix opertions like addition , subtraction , multiplication, transpose, inverse, Eigen value ,Eigen vectors and Trace calculation where a user can give input in two forms 
1) User can draw a matrix manually then take latex code from website and paste in our GUI and then tell our GUI which operations he want to perform based on above and then it will give result based on the input and operation
2) Here user can enter matrix from his/her keyboard and then tell gui which operation he want.
3) This option is not added yet where User can input image of matrix and it performs required operation as matrix images are very less in internet hence our model was giving very less accuracy so we have to remove this option as augmentation will not work in this case.





We have provided .py file for our code wihch we have merged with streamlit for webpage application


For updated code these operations will come 
"""
    **Enter a LaTeX matrix operation**  
    *Supports:*  
    - **Transpose:** Tr =\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}
    - **Trace:** Trc =\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}
    - **Inverse:** inv= \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}
    - **Determinant:** det= \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}
    - **Eigenvalues:** E =\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}
    - **Eigenvectors:** X= \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}
    - **Dimension (Rank):** dim =\begin{bmatrix} 1 & 2 \\ 2 & 4 \end{bmatrix}
    - **Rank:** R =\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}
    - **LU Decomposition:** LU =\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}
    - **SVD Decomposition:** SVD =\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}
    - **Matrix Addition, Subtraction, Multiplication (supports multiple matrices)**  
      Example:  

\begin{bmatrix} 1 & 2 \end{bmatrix} + \begin{bmatrix} 3 & 4 \end{bmatrix} + \begin{bmatrix} 5 & 6 \end{bmatrix}

    """
