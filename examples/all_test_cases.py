import numpy as np
import pandas as pd


# ============================================================
# KSOM - ALL POSSIBLE 4-BIT TESTING
# ============================================================


# ============================================================
# 1. GENERATE ALL POSSIBLE 4-BIT INPUTS
# ============================================================

X = np.array([
    [int(bit) for bit in format(i, "04b")]
    for i in range(16)
], dtype=float)


# ============================================================
# 2. INITIAL WEIGHT MATRICES
# ============================================================

weight_sets = {

    "W1": np.array([
        [0.3, 0.5, 0.7, 0.2],
        [0.6, 0.5, 0.4, 0.2]
    ], dtype=float),

    "W2": np.array([
        [0.2, 0.4, 0.6, 0.1],
        [0.7, 0.6, 0.3, 0.3]
    ], dtype=float),

    "W3": np.array([
        [0.5, 0.5, 0.5, 0.5],
        [0.2, 0.2, 0.2, 0.2]
    ], dtype=float),

    "W4": np.array([
        [0.1, 0.2, 0.3, 0.4],
        [0.8, 0.7, 0.6, 0.5]
    ], dtype=float)
}


# ============================================================
# 3. LEARNING RATE TEST CASES
# ============================================================

learning_rates = [
    0.05,
    0.10,
    0.20,
    0.50,
    0.80
]


# ============================================================
# 4. EPSILON TEST CASES
# ============================================================

epsilons = [
    0.001,
    0.0001,
    0.00001
]


# ============================================================
# 5. MAX ITERATION TEST CASES
# ============================================================

max_iterations_list = [
    50,
    100
]


# ============================================================
# 6. RESULT STORAGE
# ============================================================

iteration_results = []
final_weights_results = []
cluster_results = []
summary_results = []
input_results = []


# ============================================================
# 7. SAVE INPUT DATA
# ============================================================

for input_no, x in enumerate(X, start=1):

    input_results.append({

        "Input_No": input_no,

        "X1": int(x[0]),
        "X2": int(x[1]),
        "X3": int(x[2]),
        "X4": int(x[3]),

        "Binary":
            "".join(str(int(v)) for v in x)
    })


# ============================================================
# 8. TOTAL TEST COUNTER
# ============================================================

test_no = 0


# ============================================================
# 9. RUN ALL TEST COMBINATIONS
# ============================================================

for weight_name, initial_W in weight_sets.items():

    for learning_rate in learning_rates:

        for epsilon in epsilons:

            for max_iterations in max_iterations_list:

                test_no += 1

                W = initial_W.copy()

                converged = False

                print("\n")
                print("=" * 70)
                print("TEST:", test_no)
                print("Weight:", weight_name)
                print("Learning Rate:", learning_rate)
                print("Epsilon:", epsilon)
                print("Max Iterations:", max_iterations)
                print("=" * 70)

                # ====================================================
                # TRAINING
                # ====================================================

                for iteration in range(
                    1,
                    max_iterations + 1
                ):

                    old_W = W.copy()

                    # =================================================
                    # PROCESS ALL 16 INPUTS
                    # =================================================

                    for input_no, x in enumerate(
                        X,
                        start=1
                    ):

                        # ---------------------------------------------
                        # Distance C1
                        # ---------------------------------------------

                        distance_1 = np.sqrt(
                            np.sum(
                                (x - W[0]) ** 2
                            )
                        )

                        # ---------------------------------------------
                        # Distance C2
                        # ---------------------------------------------

                        distance_2 = np.sqrt(
                            np.sum(
                                (x - W[1]) ** 2
                            )
                        )

                        # ---------------------------------------------
                        # WINNER
                        # ---------------------------------------------

                        if distance_1 < distance_2:

                            winner = 0

                        elif distance_2 < distance_1:

                            winner = 1

                        else:

                            # Tie
                            winner = 0

                        # ---------------------------------------------
                        # OLD WEIGHT
                        # ---------------------------------------------

                        old_weight = W[winner].copy()

                        # ---------------------------------------------
                        # WEIGHT UPDATE
                        #
                        # Wnew = Wold + alpha(X - Wold)
                        #

                        W[winner] = (
                            W[winner]
                            +
                            learning_rate *
                            (x - W[winner])
                        )

                        # ---------------------------------------------
                        # WEIGHT CHANGE
                        # ---------------------------------------------

                        change = np.max(
                            np.abs(
                                W[winner]
                                -
                                old_weight
                            )
                        )

                        # ---------------------------------------------
                        # SAVE COMPLETE DATA
                        # ---------------------------------------------

                        iteration_results.append({

                            "Test":
                                test_no,

                            "Weight_Set":
                                weight_name,

                            "Learning_Rate":
                                learning_rate,

                            "Epsilon":
                                epsilon,

                            "Max_Iterations":
                                max_iterations,

                            "Iteration":
                                iteration,

                            "Input_No":
                                input_no,

                            "Input_Vector":
                                "".join(
                                    str(int(v))
                                    for v in x
                                ),

                            "Distance_C1":
                                distance_1,

                            "Distance_C2":
                                distance_2,

                            "Winner":
                                "C" + str(winner + 1),

                            "Old_W1":
                                old_weight[0],

                            "Old_W2":
                                old_weight[1],

                            "Old_W3":
                                old_weight[2],

                            "Old_W4":
                                old_weight[3],

                            "New_W1":
                                W[winner][0],

                            "New_W2":
                                W[winner][1],

                            "New_W3":
                                W[winner][2],

                            "New_W4":
                                W[winner][3],

                            "Weight_Change":
                                change
                        })

                    # =================================================
                    # CHECK CONVERGENCE
                    # =================================================

                    iteration_change = np.max(
                        np.abs(
                            W - old_W
                        )
                    )

                    if iteration_change < epsilon:

                        converged = True

                        break

                # ====================================================
                # FINAL WEIGHTS
                # ====================================================

                final_weights_results.append({

                    "Test":
                        test_no,

                    "Weight_Set":
                        weight_name,

                    "Learning_Rate":
                        learning_rate,

                    "Epsilon":
                        epsilon,

                    "Max_Iterations":
                        max_iterations,

                    "Actual_Iterations":
                        iteration,

                    "Converged":
                        converged,

                    "C1_W1":
                        W[0][0],

                    "C1_W2":
                        W[0][1],

                    "C1_W3":
                        W[0][2],

                    "C1_W4":
                        W[0][3],

                    "C2_W1":
                        W[1][0],

                    "C2_W2":
                        W[1][1],

                    "C2_W3":
                        W[1][2],

                    "C2_W4":
                        W[1][3]
                })

                # ====================================================
                # FINAL CLUSTER ASSIGNMENT
                # ====================================================

                for input_no, x in enumerate(
                    X,
                    start=1
                ):

                    distance_1 = np.sqrt(
                        np.sum(
                            (x - W[0]) ** 2
                        )
                    )

                    distance_2 = np.sqrt(
                        np.sum(
                            (x - W[1]) ** 2
                        )
                    )

                    if distance_1 < distance_2:

                        cluster = "C1"

                    elif distance_2 < distance_1:

                        cluster = "C2"

                    else:

                        cluster = "C1/C2 Tie"

                    cluster_results.append({

                        "Test":
                            test_no,

                        "Weight_Set":
                            weight_name,

                        "Learning_Rate":
                            learning_rate,

                        "Epsilon":
                            epsilon,

                        "Input_No":
                            input_no,

                        "Input_Vector":
                            "".join(
                                str(int(v))
                                for v in x
                            ),

                        "Final_Distance_C1":
                            distance_1,

                        "Final_Distance_C2":
                            distance_2,

                        "Final_Cluster":
                            cluster
                    })

                # ====================================================
                # SUMMARY
                # ====================================================

                summary_results.append({

                    "Test":
                        test_no,

                    "Weight_Set":
                        weight_name,

                    "Learning_Rate":
                        learning_rate,

                    "Epsilon":
                        epsilon,

                    "Max_Iterations":
                        max_iterations,

                    "Actual_Iterations":
                        iteration,

                    "Converged":
                        converged,

                    "Final_Max_Weight_Change":
                        iteration_change,

                    "C1_W1":
                        W[0][0],

                    "C1_W2":
                        W[0][1],

                    "C1_W3":
                        W[0][2],

                    "C1_W4":
                        W[0][3],

                    "C2_W1":
                        W[1][0],

                    "C2_W2":
                        W[1][1],

                    "C2_W3":
                        W[1][2],

                    "C2_W4":
                        W[1][3]
                })


# ============================================================
# 10. CREATE DATAFRAMES
# ============================================================

input_df = pd.DataFrame(
    input_results
)

iterations_df = pd.DataFrame(
    iteration_results
)

final_weights_df = pd.DataFrame(
    final_weights_results
)

clusters_df = pd.DataFrame(
    cluster_results
)

summary_df = pd.DataFrame(
    summary_results
)


# ============================================================
# 11. ROUND NUMERIC VALUES
# ============================================================

for df in [
    iterations_df,
    final_weights_df,
    clusters_df,
    summary_df
]:

    numeric_columns = df.select_dtypes(
        include=["float64"]
    ).columns

    df[numeric_columns] = (
        df[numeric_columns]
        .round(6)
    )


# ============================================================
# 12. PARAMETERS SHEET
# ============================================================

parameters_df = pd.DataFrame({

    "Parameter": [

        "Number of Input Bits",

        "Possible Input Patterns",

        "Number of Weight Sets",

        "Learning Rates",

        "Epsilon Values",

        "Maximum Iteration Values",

        "Total Tests"
    ],

    "Value": [

        4,

        2 ** 4,

        len(weight_sets),

        ", ".join(
            str(x)
            for x in learning_rates
        ),

        ", ".join(
            str(x)
            for x in epsilons
        ),

        ", ".join(
            str(x)
            for x in max_iterations_list
        ),

        test_no
    ]
})


# ============================================================
# 13. SAVE EXCEL
# ============================================================

file_name = "KSOM_ALL_POSSIBLE_TESTS.xlsx"


with pd.ExcelWriter(
    file_name,
    engine="openpyxl"
) as writer:

    input_df.to_excel(
        writer,
        sheet_name="All 4-Bit Inputs",
        index=False
    )

    parameters_df.to_excel(
        writer,
        sheet_name="Parameters",
        index=False
    )

    iterations_df.to_excel(
        writer,
        sheet_name="Iterations",
        index=False
    )

    final_weights_df.to_excel(
        writer,
        sheet_name="Final Weights",
        index=False
    )

    clusters_df.to_excel(
        writer,
        sheet_name="Clusters",
        index=False
    )

    summary_df.to_excel(
        writer,
        sheet_name="Summary",
        index=False
    )


# ============================================================
# 14. FINAL OUTPUT
# ============================================================

print("\n")
print("=" * 70)
print("KSOM COMPLETE TESTING FINISHED")
print("=" * 70)

print("\nTotal possible 4-bit inputs:", 16)

print(
    "Weight sets:",
    len(weight_sets)
)

print(
    "Learning rates:",
    len(learning_rates)
)

print(
    "Epsilon values:",
    len(epsilons)
)

print(
    "Max iteration values:",
    len(max_iterations_list)
)

print(
    "\nTOTAL TEST CASES:",
    test_no
)

print(
    "\nExcel file:",
    file_name
)

print("\n")
print("=" * 70)
print("SUMMARY")
print("=" * 70)

print(
    summary_df.to_string(
        index=False
    )
)
