# Just in Time or Just a Guess? - Shiny App

This Shiny app accompanies the paper *"Just in Time or Just a Guess? Addressing Challenges in Validating Prediction Models Based on Longitudinal Data"* (in prep, Langener & Jacobson, 2025).

## Run the App Online

If you have a smaller dataset, you can run the app directly in your browser:\
<https://annalangener.shinyapps.io/Justintime/>

## Run the App Locally

To run the app on your local machine, follow these steps;

1.  Make sure you have R installed. If not, you can download it here: <https://cran.rstudio.com/>

2.  Run the following code in the **terminal**;

    ```{r}
    if (!requireNamespace("shiny", quietly = TRUE)) {
      install.packages("shiny")
    }

    library("shiny")

    runGitHub("AnnaLangener/Justintime")
    ```

## Overview of important files

-   **app.R**: Contains the Shiny app code.
-   **Simulation_Functions.R**: Includes functions to run custom simulations (Tab: *Create Simulated Data*).
-   **Simulation_UploadData.R**: Includes functions to apply cross-validation strategies to uploaded datasets (Tab: *Upload Your Own Data*).
-   **exampledata_high.csv**: Simulated example dataset with high variance explained by between-person differences.
-   **exampledata_low.csv**: Simulated example dataset with low variance explained by between-person differences.
