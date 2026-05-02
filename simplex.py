"""
Implementation of the simplex algorithm for solving linear programming problems.
The simplex algorithm is an iterative method for finding the optimal solution of a linear programming problem by moving along the edges of the feasible region defined by the constraints. 
The algorithm starts with an initial basic feasible solution and iteratively improves the solution by performing pivot operations until an optimal solution is found or it is determined that the problem is unbounded or infeasible. 
This implementation assumes that the input linear programming problem is in standard form, where the objective function is to be maximized, all constraints are in the form of less than or equal to inequalities, and all variables are non-negative.
"""



import numpy as np
import streamlit as st
import pandas as pd
import io

class simplex():
    def __init__(self, file):
        """
        Create an instance of the simplex class by reading the input file. The input file should be in the following format:
        - First line: m n (number of constraints and number of variables)
        - Next m lines: coefficients of the constraints (A matrix) and the constants (b vector)
        - Last line: coefficients of the objective function (c vector)

        param file: path to the input file
        type file: str, required

        return: model instance of the simplex class
        rtype: simplex model
        """    
        self.vars = []

        file.seek(0)
        f = io.TextIOWrapper(file, encoding='utf-8')

        line = const = n_var = 0 #const = number of constraints, n_var = number of variables
        for data in f:
            if line == 0: #Extract matrix size and initialize A, b and c
                print("Header found!")
                data = data.split()
                
                self.A = np.zeros((int(data[0]), int(data[1])))
                self.b = np.zeros(int(data[0]))
                self.c = np.zeros(int(data[1]))
                self.const_type = []
                self.optimity = None
                self.optimity_original = None

                const = int(data[0])
                n_var = int(data[1])
                
                for var in range(n_var):
                    self.addVar("x", "x"+str(n_var+1))
            
            if line != 0 and line <= const: #Extract constraints
                self.addConst(data, line) #Please note that line number starts from 1, not 0
            
            if line > const: #Extract objective function
                self.setObjective(data)
            line += 1
        
    def addVar(self, type, name):
        """
        Add a variable to the model. This function is used to keep track of the variables in the model, but it does not affect the simplex algorithm itself.

        :param type: currently not in use
        :param name: name of the variable
        :type type: str, required
        """
        self.vars.append((type, name))

    def addConst(self, data, number: int):
        """
        Add a constraint to the model. This function is used to populate the A matrix and b vector of the model.

        :param data: coefficients of the constraint and the constant
        :type data: str, required
        :param number: line number of the constraint in the input file (starting from 1)
        :type number: int, required
        """
        data = data.split()
                
        for d, var in zip(data, range(len(self.vars)+2)):
            if var < len(self.vars):
                self.A[number-1][var] = float(d)
            elif var == len(self.vars):
                self.const_type.append(d)
            else:
                self.b[number-1] = float(d)
        if self.const_type[-1] == ">=":
            self.A[number-1] = (-1) * self.A[number-1] #Multiply the coefficients of the variables in the constraint by -1 to convert the constraint to a <= constraint for the simplex algorithm
            self.b[number-1] = (-1) * self.b[number-1] #Multiply the constant term of the constraint by -1
            self.const_type[-1] = "<=" #Update the constraint type to <= for the simplex algorithm
            st.write(f"Constraint {number} converted to <= constraint for the simplex algorithm: {self.A[number-1]} <= {self.b[number-1]}")

    def setObjective(self, data):
        """
        Add the objective function to the model. This function is used to populate the c vector of the model.

        :param data: coefficients of the objective function
        :type data: str, required
        """
        data = data.split()
        self.optimity = data[0]
        self.optimity_original = data[0]
        if self.optimity == "max":
            self.c = np.array([float(d) for d in data[1:]])
        elif self.optimity == "min":
            self.c = np.array([(-1) * float(d) for d in data[1:]]) #Multiply by -1 to convert the minimization problem to a maximization problem for the simplex algorithm
            self.optimity = "max" #Update the optimity to max for the simplex algorithm
        else:
            raise Exception("Invalid mode! Mode should be either 'max' or  d'min'.")

    def __init_simplex_elements(self):
        """
        Initialize the simplex dictionary and the objective function for the simplex algorithm. 
        The simplex dictionary is a matrix that represents the current solution of the simplex algorithm, and it is initialized with the coefficients of the constraints and the constants. 
        The objective function is initialized with the coefficients of the objective function and a constant term of 0. 
        The basis and non-basis variables are also initialized to keep track of which variables are currently in the basis and which are not.

        Convention:
        - Basis variables are denoted as s1, s2, ..., sm (slack variables)
        - Non-basis variables are denoted as x1, x2, ..., xn (original variables)
        - dict: A matrix where each row corresponds to a constraint and each column corresponds to a variable (including the constant term). 
        The first column of dict represents the constant term of the constraints, and the remaining columns represent the coefficients of the variables in the constraints. 
        The coefficients are negated to facilitate the simplex algorithm's pivot operations. 
        - simplex_objective: A vector that represents the coefficients of the objective function, where the first element is the constant term (initialized to 0) and the remaining elements are the coefficients of the original variables (c vector).
        """
        self.dict = np.zeros((self.A.shape[0], self.A.shape[1]+1))
        self.basis = ["s" + str(i) for i in range(self.A.shape[0])]
        self.non_basis = ["x" + str(i) for i in range(self.A.shape[1])]
        self.simplex_objective = np.zeros(self.A.shape[1]+1)
        self.s = np.zeros(self.A.shape[0])
        
        for i in range(self.A.shape[0]):
            self.s[i] = self.b[i]

        for i in range(self.A.shape[0]):
            for j in range(self.A.shape[1]+1):
                if j == 0: #Case for the constant term of the constraints
                    self.dict[i][j] = self.b[i]
                else: #Case for the coefficients of the variables in the constraints multiplied by -1 for the convention: basis_variables = constant_term - sum(non_basis_variables * coefficients)
                    self.dict[i][j] = (-1) * self.A[i][j-1]

        self.simplex_objective[0] = 0
        for i in range(self.c.shape[0]):
            self.simplex_objective[i+1] = self.c[i]

    def __find_pivot(self):
        """
        Find the pivot column for the simplex algorithm. 
        The pivot column is the first column with a positive coefficient in the objective function, which indicates the variable that will enter the basis in the next iteration of the simplex algorithm.
        
        :return: index of the pivot column
        :rtype: int
        """
        pivot = None
        current_val = 0
        for i in range(1, self.simplex_objective.shape[0]):
            if self.simplex_objective[i] > 0 and self.simplex_objective[i] > current_val:
                current_val = self.simplex_objective[i]
                pivot = i - 1
            
        if pivot is not None:
            return pivot
        raise Exception("Optimal solution found!")
            
    def __find_non_basis_variable(self, pivot):
        # Wir suchen die Zeile i, die den kleinsten positiven Quotienten liefert
        min_ratio = float('inf')
        leaving_row_index = -1

        # In deinem Dictionary: Spalte 0 ist b, Spalte pivot+1 ist der Koeffizient
        for i in range(self.dict.shape[0]):
            coeff = self.dict[i][pivot + 1]
            constant = self.dict[i][0]

            # Begrenzung findet nur statt, wenn der Koeffizient negativ ist 
            # (da Basis = b - (-coeff * x) -> Basis = b + coeff * x würde nicht begrenzen)
            # Wenn im Dict -4 steht, ist die Gleichung: Basis = 10 - 4x.
            if coeff < 0:
                # Quotient = b / |coeff|
                ratio = constant / abs(coeff)
                
                if ratio < min_ratio:
                    min_ratio = ratio
                    leaving_row_index = i

        # Wenn kein Koeffizient negativ war, gibt es keine Schranke
        if leaving_row_index == -1:
            return -1 # Unbounded

        return leaving_row_index
    
    def __update_simplex(self, pivot, pivot_max):
        """
        Update the simplex dictionary and the objective function for the next iteration of the simplex algorithm.
        The simplex dictionary is updated by performing row operations to make the pivot column of the pivot row have a 1 in the pivot position and 0s elsewhere.

        :param pivot: index of the pivot column
        :type pivot: int, required
        :param pivot_max: index of the variable that will leave the basis
        :type pivot_max: int, required
        """

        #First of all, we need to update the basis and non-basis variables by swapping the variable that will enter the basis (pivot column) with the variable that will leave the basis (pivot row) 
        temp = self.basis[pivot_max]
        self.basis[pivot_max] = self.non_basis[pivot]
        self.non_basis[pivot] = temp

        #Create a temporary copy of the simplex dictionary to perform the row operations without affecting the original dictionary until all updates are done
        temp_dict = self.dict.copy()

        #First update step is to make the pivot column of the pivot row have a 1 in the pivot position by dividing the entire pivot row by the coefficient of the pivot column in that row
        for i in range(self.dict.shape[1]): #Update pivot row
            if i != pivot+1:
                temp_dict[pivot_max][i] = (-1) * self.dict[pivot_max][i] / self.dict[pivot_max][pivot+1] #Multiply by -1 to maintain the convention of my implementation
            else:
                temp_dict[pivot_max][i] = 1 / self.dict[pivot_max][pivot+1] #Set the pivot column of the pivot row to 1
        
        #Second update step is to make the pivot column of the other rows
        for i in range(self.dict.shape[0]): #Update other rows
            if i != pivot_max:
                for j in range(self.dict.shape[1]):
                    if j == pivot+1: #Set the pivot column of the other rows to 0 by performing row operations
                        temp_dict[i][j] = 0
                    temp_dict[i][j] += self.dict[i][pivot+1] * temp_dict[pivot_max][j]

        self.dict = temp_dict

        #Last update step is to update the objective function by performing row operations to make the coefficient of the pivot column in the objective function and updating the other coefficients accordingly
        temp_objective = self.simplex_objective.copy()
        for i in range(self.simplex_objective.shape[0]):
            if i == pivot+1:
                self.simplex_objective[i] = 0
            self.simplex_objective[i] += temp_objective[pivot+1] * temp_dict[pivot_max][i]

    def __print_simplex_iteration(self, pivot, pivot_constr):
        """
        Print the current iteration of the simplex algorithm.

        :param pivot: index of the pivot column
        :type pivot: int, required
        :param pivot_constr: index of the variable that will leave the basis
        :type pivot_constr: int, required
        """
        # 1. Pivot information
        st.subheader("Pivot Selection")
        col_a, col_b = st.columns(2)
        with col_a:
            st.info(f"**Pivot Column:** {self.non_basis[pivot]}")
        with col_b:
            st.info(f"**Pivot Row:** {self.basis[pivot_constr]}")

        # 2. Display the Simplex Dictionary as a DataFrame with styling
        st.subheader("Simplex Tableau")
        
        # Create a DataFrame for display
        columns = ['Constant'] + self.non_basis
        df = pd.DataFrame(self.dict, index=self.basis, columns=columns)
        
        # Styling: Pivot row and pivot column highlighted with a light orange background
        def highlight_pivot(x):
            df_color = pd.DataFrame('', index=x.index, columns=x.columns)
            # Highlight the pivot row
            df_color.iloc[pivot_constr, :] = 'background-color: rgba(255, 165, 0, 0.2)' 
            # Highlight the pivot column
            df_color.iloc[:, pivot + 1] = 'background-color: rgba(255, 165, 0, 0.2)'
            return df_color

        st.dataframe(df.style.format("{:.3f}").apply(highlight_pivot, axis=None), use_container_width=True)

        # 3. Linear form
        with st.expander("Show Linear Form"):
            for i, row_var in enumerate(self.basis):
                eq = rf"{row_var} = {self.dict[i][0]:.3f}"
                for j, col_var in enumerate(self.non_basis):
                    val = self.dict[i][j+1]
                    sign = "+" if val >= 0 else "-"
                    eq += f" {sign} {abs(val):.3f} \cdot {col_var}"
                st.latex(eq)

        # 4. Objective function
        st.divider()
        
        # Objective function in LaTeX
        obj_latex = rf"z = {self.simplex_objective[0]:.3f}"
        for i, var in enumerate(self.non_basis):
            val = self.simplex_objective[i+1]
            sign = "+" if val >= 0 else "-"
            obj_latex += f" {sign} {abs(val):.3f} \cdot {var}"
        
        st.write("**Current Objective Function:**")
        st.latex(obj_latex)
        
        # The current value of the objective function (z) is displayed as a metric for better visibility
        st.metric(label="Current Objective Value (z)", value=f"{self.simplex_objective[0]:.3f}")

    def __check_negativ_b(self, constraint, b):
        """
        Check if the constant term of the constraint is negative. This is used to determine if we need to perform the two phase simplex method to find a feasible solution for the original problem.

        :param constraint: index of the constraint to check
        :type constraint: int, required

        :param b: array of constant terms
        :type b: np.ndarray, required

        :return: True if the constant term of the constraint is negative, False otherwise
        :rtype: bool
        """
        return b[constraint] < 0

    def __init_two_phase_simplex_elements(self):
        self.dict = np.zeros((self.A.shape[0], self.A.shape[1]+1+self.A.shape[0])) #+2 = Additional column for the artificial variables
        self.basis = []
        self.non_basis = ["x" + str(i) for i in range(self.A.shape[1])] #Original variables are non-basis variables at the beginning of the two phase simplex method
        self.simplex_objective = np.zeros(self.dict.shape[1]) #Objective function for the two phase simplex method, where the coefficients of the original variables are 0 and the coefficients of the artificial variables are 1
        self.s = np.zeros(self.A.shape[0])
        self.a = np.zeros(self.A.shape[0]) #Artificial variables for two phase simplex method

        for i in range(self.A.shape[0]):
            self.a[i] = self.b[i]
            self.s[i] = 0

        A = self.A.copy()
        b = self.b.copy()

        i = 0
        while i < self.A.shape[0]:
            if self.__check_negativ_b(i, b):
                A[i] = (-1) * A[i] #Multiply the coefficients of the variables in the constraint by -1 to convert the constraint to a >= constraint for the two phase simplex method
                b[i] = (-1) * b[i] #Multiply the constant term of the constraint by -1 
                if self.const_type[i] == ">=":
                    self.const_type[i] = "<=" #Update the constraint type to <= for the two phase simplex method
                elif self.const_type[i] == "<=":
                    self.const_type[i] = ">=" #Update the constraint type to >= for the two phase simplex method
                continue
            
            if self.const_type[i] == ">=":
                for j in range(self.A.shape[1]+1+self.A.shape[0]):
                    if j == 0: #Case for the constant term of the constraints
                        self.dict[i][j] = b[i]
                    elif j <= self.A.shape[1]: #Case for the coefficients of the variables in the constraints multiplied by -1 for the convention: basis_variables = constant_term - sum(non_basis_variables * coefficients)
                        self.dict[i][j] = (-1) * A[i][j-1]
                    else:
                        if j == self.A.shape[1]+1+i:
                            self.dict[i][j] = 1
                            if "s" + str(i) not in self.non_basis:
                                self.non_basis.append("s" + str(i)) #Set the slack variable as a non-basis variable for the corresponding constraint if it is not already in the non-basis variables (in case of >= constraints)
                        else:
                            self.dict[i][j] = 0
                self.basis.append("a" + str(i)) #Set the artificial variable as the basis variable for the corresponding constraint
            
            elif self.const_type[i] == "<=":
                for j in range(self.A.shape[1]+1+self.A.shape[0]):
                    if j == 0: #Case for the constant term of the constraints
                        self.dict[i][j] = b[i]
                    elif j <= self.A.shape[1]: #Case for the coefficients of the variables in the constraints multiplied by -1 for the convention: basis_variables = constant_term - sum(non_basis_variables * coefficients)
                        self.dict[i][j] = (-1) * A[i][j-1]
                    else:
                        self.dict[i][j] = 0 #Set the coefficients of the artificial variables in the constraints to 0 for <= constraints, since we only need artificial variables for >= constraints in the two phase simplex method
                self.basis.append("s" + str(i)) #Set the slack variable as the basis variable for the corresponding constraint
            
            elif self.const_type[i] == "=":
                for j in range(self.A.shape[1]+1+self.A.shape[0]):
                    if j == 0: #Case for the constant term of the constraints
                        self.dict[i][j] = b[i]
                    elif j <= self.A.shape[1]: #Case for the coefficients of the variables in the constraints multiplied by -1 for the convention: basis_variables = constant_term - sum(non_basis_variables * coefficients)
                        self.dict[i][j] = (-1) * A[i][j-1]
                    else:
                        self.dict[i][j] = 0
                self.basis.append("a" + str(i)) #Set the artificial variable as the basis variable for the corresponding constraint
            else:
                raise Exception("Invalid constraint type! Constraint type should be either '<=', '>=', or '='.")
            
            i += 1

        target_cols = 1 + len(self.non_basis)  # Konstante + alle Non-Basis-Spalten
        if self.dict.shape[1] > target_cols:
            cols_to_delete = np.arange(target_cols, self.dict.shape[1])
            self.dict = np.delete(self.dict, cols_to_delete, axis=1)
            self.simplex_objective = np.delete(self.simplex_objective, cols_to_delete)
        
        for i in range(self.A.shape[0]):
            if self.basis[i].startswith("a"):  # Only rows with artificial variables
                # We subtract the row because: max w = sum(-a_i)
                # and in dict: a_i = b_i - coeff*x -> -a_i = -b_i + coeff*x
                # Since we are maximizing, we need to adjust the coefficients
                # so that they correctly reflect the rows.
                self.simplex_objective -= self.dict[i]

        st.subheader("Two-Phase Simplex Starting Tableau")
        
        # Create a DataFrame for display
        columns = ['Constant'] + self.non_basis
        df = pd.DataFrame(self.dict, index=self.basis, columns=columns)
        st.dataframe(df.style.format("{:.3f}"), use_container_width=True)

    def __transport_into_phase_2(self, old_dict, old_basis, old_non_basis):
        """
        Transport the tableau from phase 1 to phase 2.

        - Removes artificial variables from the non-basis variables
        - Reconstructs the tableau only with the remaining variables
        - Correctly builds the objective function of the original problem
        """
        # Only keep variables that are not artificial
        new_non_basis = [var for var in old_non_basis if not var.startswith("a")]

        # Mapping: variable name -> column index in old tableau
        old_var_to_col = {var: idx + 1 for idx, var in enumerate(old_non_basis)}

        # New tableau: constant + all remaining non-basis variables
        new_dict = np.zeros((old_dict.shape[0], len(new_non_basis) + 1))
        new_dict[:, 0] = old_dict[:, 0]

        for new_idx, var in enumerate(new_non_basis, start=1):
            new_dict[:, new_idx] = old_dict[:, old_var_to_col[var]]

        # Rebuild the objective function for the original problem:
        # z = c^T x
        new_simplex_objective = np.zeros(len(new_non_basis) + 1)

        # 1) Contributions from current non-basis variables
        for idx, var in enumerate(new_non_basis, start=1):
            if var.startswith("x"):
                var_num = int(var[1:])
                # Use the costs from the original problem (self.c)
                new_simplex_objective[idx] = self.c[var_num]

        # 2) Include contributions from basis variables
        #    If x_j is in the basis: x_j = b + sum(coeff_k * y_k)
        #    => z += c_j * row
        for row_idx, basis_var in enumerate(old_basis):
            if basis_var.startswith("x"):
                var_num = int(basis_var[1:])
                coeff = self.c[var_num]
                if coeff != 0:
                    new_simplex_objective += coeff * new_dict[row_idx, :]

        self.dict = new_dict
        self.non_basis = new_non_basis
        self.basis = old_basis.copy()
        self.simplex_objective = new_simplex_objective.copy()

        st.subheader("Transported Simplex Starting Tableau")
        
        # Create a DataFrame for display
        columns = ['Constant'] + self.non_basis
        df = pd.DataFrame(self.dict, index=self.basis, columns=columns)
        st.dataframe(df.style.format("{:.3f}"), use_container_width=True)

        # Objective function in LaTeX
        st.divider() 
        obj_latex = rf"z = {self.simplex_objective[0]:.3f}"
        for i, var in enumerate(self.non_basis):
            val = self.simplex_objective[i+1]
            sign = "+" if val >= 0 else "-"
            obj_latex += f" {sign} {abs(val):.3f} \cdot {var}"
        
        st.write("**Transported Objective Function:**")
        st.latex(obj_latex)

    def __two_phase_simplex(self):
        st.header("Two-Phase Simplex")
        st.info("Starting two-phase simplex method to find a feasible solution for the original problem.")
        self.__init_two_phase_simplex_elements()
        st.info("Phase 1: Two-Phase Simplex Method - Finding a Feasible Solution for the Original Problem")
        itr = 1  # Number of iterations
        while True:

            pivot = None
            try:
                pivot = self.__find_pivot()  # In the first phase of the two-phase simplex method, we want to minimize the sum of the artificial variables to find a feasible solution for the original problem. Therefore, we need to find the pivot column with the most negative coefficient in the objective function.
            except Exception as e:
                print(self.simplex_objective[0])
                st.info(str(e))
                break
            
            pivot_constr = self.__find_non_basis_variable(pivot)

            if pivot_constr == -1:
                st.error("The problem is unbounded!")
                self.objective_value = None
                break

            self.__update_simplex(pivot, pivot_constr)

            self.__print_simplex_iteration(pivot, pivot_constr)

            itr += 1

        if abs(self.simplex_objective[0]) > 1e-6:  # If the optimal value of the objective function in the first phase of the two-phase simplex method is not 0, then the original problem is infeasible, since we want to minimize the sum of the artificial variables to find a feasible solution for the original problem. Therefore, if the optimal value of the objective function in the first phase is not 0, it means that at least one artificial variable is still in the basis with a positive value, which indicates that there is no feasible solution for the original problem.
            st.error("The original problem is infeasible! No feasible solution found.")
            self.objective_value = None
            return -1
        else:
            st.success("The original problem is feasible! Starting simplex algorithm for the original problem.")
            st.info("Phase 2: Simplex Method for the Original Problem - Finding the Optimal Solution for the Original Problem")
            self.__transport_into_phase_2(self.dict, self.basis, self.non_basis)  # Transport the simplex dictionary and the basis variables from the first phase of the two-phase simplex method to the simplex algorithm for the original problem to continue with the simplex algorithm for the original problem after we have found a feasible solution for the original problem with the two-phase simplex method.
            return 0
    
    
    def solve(self):
        """
        Perform the simplex algorithm to find the optimal solution of the linear programming problem.
        """
        if (self.b >= 0).all():
            self.__init_simplex_elements()  # Initialize dict with starting parameters
        else:
            st.warning("Negative constants detected! Starting two-phase simplex method to find a feasible solution for the original problem.")
            simplex_const_type = self.const_type.copy()  # Copy the original constraint types to keep track of them during the two-phase simplex method
            check_value_for_infeasible = self.__two_phase_simplex()  # Initialize dict with starting parameters and additional artificial variables for the two-phase simplex method to find a feasible solution for the original problem.
            self.const_type = simplex_const_type.copy()  # Restore the original constraint types after the two-phase simplex method is done to continue with the simplex algorithm for the original problem
        
            if check_value_for_infeasible == -1:  # If the original problem is infeasible, we can stop here and return None as the optimal solution value, since there is no feasible solution for the original problem.
                return None
        
        st.header("Simplex Algorithm")
        itr = 1  # Number of iterations
        while True:
            st.write(f"----------------------- Iteration {itr} -----------------------")

            pivot = None
            try:
                pivot = self.__find_pivot()
            except Exception as e:
                st.success(str(e))
                self.objective_value = self.simplex_objective[0]
                st.success(f"Optimal Solution Value: {self.objective_value}")
                break
            
            pivot_constr = self.__find_non_basis_variable(pivot)

            if pivot_constr == -1:
                st.error("The problem is unbounded!")
                self.objective_value = None
                break

            self.__update_simplex(pivot, pivot_constr)

            self.__print_simplex_iteration(pivot, pivot_constr)

            itr += 1
                


st.title("Simplex Algorithm for Linear Programming")

# File uploader
uploaded_file = st.file_uploader("Choose a TXT file", type=["txt"])

if uploaded_file is not None:
    # Read file (Streamlit treats the uploaded object like a file)
    if uploaded_file.name.endswith('.txt'):
        df = pd.read_csv(uploaded_file, delimiter='\t')
    else:
        df = pd.read_excel(uploaded_file)

    st.success("File loaded successfully!")
    
    # Display table
    st.write("Data preview:")
    st.dataframe(df)

    model = simplex(uploaded_file)
    model.solve()

else:
    st.info("Please upload a TXT file.")