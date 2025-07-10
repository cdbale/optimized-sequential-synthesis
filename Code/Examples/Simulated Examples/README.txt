This example simulates a series of datasets meant to represent churn in a series of individuals exposed to advertising for a brand.

In these examples, the respondents have three explanatory variables that the company may have: age, num_visits (meant to represent the number of times an individual has visited the brand's website in the past), and amount_spent (the amount an individual has spent on the brand so far). Additionally, there are three "Affinity Attributes" that a third-party provider such as Google may have based on the individual's internet activity: whether or not the individual is interested in hiking, their interest in sustainability, and whether or not the individual is highly active online. Finally, the outcome variable is whether or not the individual has churned.

Three data simulations have been set up, each with different levels of dependence for both churn and the affinity attributes. These are meant to simulate both the different relationships that may occur within the data as well as the different levels of information that may be shared between a brand and the walled garden within which the brand is advertising.

Once the data is simulated in the "Simulate Data" R scripts, the "Synthesize Data" python script is run. These scripts perform the synthesis as outlined in the paper. Then, the "Regressions" R scripts perform model comparisons on logistic regressions of both the simulated data and the synthetic data, as well as on different amounts of information that are provided to the model.

The "Compare All Simulated Models" runs all of the previous R scripts given the simulated and synthesized data to provide a a simple visualization of the effectiveness of all the different models.
