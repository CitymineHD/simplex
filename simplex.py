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
            elif var > len(self.vars):
                    self.b[number-1] = float(d)

    def setObjective(self, data):
        """
        Add the objective function to the model. This function is used to populate the c vector of the model.

        :param data: coefficients of the objective function
        :type data: str, required
        """
        data = data.split()
        self.c = np.array([float(d) for d in data])

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

    def __find_pivot(self, mode="max"):
        """
        Find the pivot column for the simplex algorithm. 
        The pivot column is the first column with a positive coefficient in the objective function, which indicates the variable that will enter the basis in the next iteration of the simplex algorithm.
        
        :return: index of the pivot column
        :rtype: int
        """
        for i in range(1, self.simplex_objective.shape[0]):
            if mode == "max":
                if self.simplex_objective[i] > 0:
                    return i-1 #Return the index of the non-basis variable that corresponds to the pivot column (subtract 1 because the first element of simplex_objective is the constant term)
            elif mode == "min":
                if self.simplex_objective[i] < 0:
                    return i-1 #Return the index of the non-basis variable that corresponds to the pivot column (subtract 1 because the first element of simplex_objective is the constant term)
            else:
                raise Exception("Invalid mode! Mode should be either 'max' or 'min'.")
        raise Exception("Optimal solution found!")
            
    def __find_non_basis_variable(self, pivot):
        """
        First determine if the problem is unbounded by checking if all coefficients in the pivot column of the simplex dictionary are non-negative. 
        If they are, then the problem is unbounded and we can return -1 to indicate this.
        Find the variable that will leave the basis in the next iteration of the simplex algorithm.
        The variable that will leave the basis is determined by the minimum ratio test, which is calculated by dividing the constant term of each constraint by the coefficient of the pivot column in that constraint.

        :param pivot: index of the pivot column
        :type pivot: int, required

        :return: -1 if the problem is unbounded, otherwise the index of the variable that will leave the basis
        :rtype: int
        """

        if (self.dict[:, pivot+1] >= 0).all():
            return -1

        max = index = 0
        for i in range(self.dict.shape[0]):
            if self.dict[i][pivot+1] < 0:
                ratio = (-1) * self.dict[i][0] / self.dict[i][pivot+1] #Calculate the ratio by dividing the constant term of the constraint by the coefficient of the pivot column in that constraint (multiply by -1 because the coefficients in the simplex dictionary are negated)
                if ratio < max or max == 0:
                    max = ratio
                    index = i

        return index
    
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
        # 1. Pivot-Informationen
        st.subheader("Pivot-Auswahl")
        col_a, col_b = st.columns(2)
        with col_a:
            st.info(f"**Pivot-Spalte:** {self.non_basis[pivot]}")
        with col_b:
            st.info(f"**Pivot-Zeile:** {self.basis[pivot_constr]}")

        # 2. Display the Simplex Dictionary as a DataFrame with styling
        st.subheader("Simplex-Tableau")
        
        # Erstellen eines DataFrames für die Darstellung
        columns = ['Konstante'] + self.non_basis
        df = pd.DataFrame(self.dict, index=self.basis, columns=columns)
        
        # Styling: Pivot-Row and Pivot-Column highlighted with a light orange background
        def highlight_pivot(x):
            df_color = pd.DataFrame('', index=x.index, columns=x.columns)
            # Markiere die Pivot-Zeile
            df_color.iloc[pivot_constr, :] = 'background-color: rgba(255, 165, 0, 0.2)' 
            # Markiere die Pivot-Spalte
            df_color.iloc[:, pivot + 1] = 'background-color: rgba(255, 165, 0, 0.2)'
            return df_color

        st.dataframe(df.style.format("{:.3f}").apply(highlight_pivot, axis=None), use_container_width=True)

        # 3. Linear Form
        with st.expander("Lineare Form anzeigen"):
            for i, row_var in enumerate(self.basis):
                eq = rf"{row_var} = {self.dict[i][0]:.3f}"
                for j, col_var in enumerate(self.non_basis):
                    val = self.dict[i][j+1]
                    sign = "+" if val >= 0 else "-"
                    eq += f" {sign} {abs(val):.3f} \cdot {col_var}"
                st.latex(eq)

        # 4. Objective Function
        st.divider()
        
        # Objective Function in LaTeX
        obj_latex = rf"z = {self.simplex_objective[0]:.3f}"
        for i, var in enumerate(self.non_basis):
            val = self.simplex_objective[i+1]
            sign = "+" if val >= 0 else "-"
            obj_latex += f" {sign} {abs(val):.3f} \cdot {var}"
        
        st.write("**Aktuelle Zielfunktion:**")
        st.latex(obj_latex)
        
        # The current value of the objective function (z) is displayed as a metric for better visibility
        st.metric(label="Aktueller Zielwert (z)", value=f"{self.simplex_objective[0]:.3f}")

    def __init_two_phase_simplex_elements(self):
        self.dict = np.zeros((self.A.shape[0], self.A.shape[1]+1+self.A.shape[0])) #+2 = Additional column for the artificial variables
        self.basis = ["a" + str(i) for i in range(self.A.shape[0])]
        self.non_basis = ["x" + str(i) for i in range(self.A.shape[1])] + ["s" + str(i) for i in range(self.A.shape[0])]
        self.simplex_objective = np.ones(self.A.shape[0]+1)
        self.s = np.zeros(self.A.shape[0])
        self.a = np.zeros(self.A.shape[0]) #Artificial variables for two phase simplex method

        for i in range(self.A.shape[0]):
            self.a[i] = self.b[i]

        for i in range(self.A.shape[0]):
            for j in range(self.A.shape[1]+1):
                if j == 0: #Case for the constant term of the constraints
                    self.dict[i][j] = self.b[i]
                else: #Case for the coefficients of the variables in the constraints multiplied by -1 for the convention: basis_variables = constant_term - sum(non_basis_variables * coefficients)
                    self.dict[i][j] = (-1) * self.A[i][j-1]

        for i in range(self.A.shape[0]):
            self.dict[i][self.A.shape[1]+1+i] = 1 #Set the coefficients of the artificial variables to 1 in the simplex dictionary 

        self.simplex_objective[0] = 0

    def __two_phase_simplex(self):
        self.__init_two_phase_simplex_elements()
        itr = 1 #Number of Interations
        while True:
            #st.write(f"----------------------- Iteration {itr} -----------------------")

            pivot = None
            try:
                pivot = self.__find_pivot(mode="min") #In the first phase of the two phase simplex method, we want to minimize the sum of the artificial variables to find a feasible solution for the original problem. Therefore, we need to find the pivot column with the most negative coefficient in the objective function.
            except Exception as e:
                print(self.simplex_objective[0])
                break
            
            print(f"Pivot column: {self.non_basis[pivot]}")
            pivot_constr = self.__find_non_basis_variable(pivot)

            print(f"Pivot row: {self.basis[pivot_constr]}")
            if pivot_constr == -1:
                print("The problem is unbounded!")
                self.objective_value = None
                break

            self.__update_simplex(pivot, pivot_constr)

            print(self.simplex_objective)

            #self.__print_simplex_iteration(pivot, pivot_constr)

            itr += 1


    def solve(self):
        """
        Perfom the simplex algorithm to find the optimal solution of the linear programming problem.
        """
        if (self.b >= 0).all():
            self.__init_simplex_elements() #Init Dict with starting parmeters = 0
        else:
            st.warning("Negative constants detected! Starting two phase simplex method to find a feasible solution for the original problem.")
            self.__two_phase_simplex() #Init Dict with starting parmeters = 0 and additional artificial variables for the two phase simplex method to find a feasible solution for the original problem.
            return None;
        
        itr = 1 #Number of Interations
        while True:
            st.write(f"----------------------- Iteration {itr} -----------------------")

            pivot = None
            try:
                pivot = self.__find_pivot()
            except Exception as e:
                st.write(str(e))
                self.objective_value = self.simplex_objective[0]
                st.write(f"Optimal solution Value: {self.objective_value}")
                break
            
            pivot_constr = self.__find_non_basis_variable(pivot)

            if pivot_constr == -1:
                st.write("The problem is unbounded!")
                self.objective_value = None
                break

            self.__update_simplex(pivot, pivot_constr)

            self.__print_simplex_iteration(pivot, pivot_constr)

            itr += 1
                



st.title("Datei-Import Test")

# Der File Uploader
uploaded_file = st.file_uploader("Wähle eine TXT-Datei aus", type=["txt"])

if uploaded_file is not None:
    # Datei einlesen (Streamlit behandelt das hochgeladene Objekt wie eine Datei)
    if uploaded_file.name.endswith('.txt'):
        df = pd.read_csv(uploaded_file, delimiter='\t')
    else:
        df = pd.read_excel(uploaded_file)

    st.success("Datei erfolgreich geladen!")
    
    # Tabelle anzeigen
    st.write("Vorschau der Daten:")
    st.dataframe(df)

    model = simplex(uploaded_file)
    model.solve()

else:
    st.info("Bitte lade eine TXT-Datei hoch.")