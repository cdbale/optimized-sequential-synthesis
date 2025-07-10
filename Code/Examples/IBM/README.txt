This folder contains the simulation, synthesis, and model comparison for IBM Telco's data. The following is IBM's summary of the dataset (a full description can be found at https://community.ibm.com/community/user/blogs/steven-macko/2019/07/11/telco-customer-churn-1113):

"The Telco customer churn data contains information about a fictional telco company that provided home phone and Internet services to 7043 customers in California in Q3. It indicates which customers have left, stayed, or signed up for their service. Multiple important demographics are included for each customer, as well as a Satisfaction Score, Churn Score, and Customer Lifetime Value (CLTV) index."

In this example, a third-party company is attempting to ascertain churn while also dealing with certain variables that are obscured due to privacy restrictions. These variables involve the personal aspects of an individual's internet habits, such as whether they stream movies, TV, or the details of their internet plan.

The "IBM Telco Cleaning" R script formats the data so that it can be read by the "Synthesize IBM Data" python script. Both datasets are then imported into the "IBM Model Comparison" R script to determine whether the model using partial data, the model using fully synthetic data, or the model using the complete original dataset Is most effective at determining churn.

WARNING: This example was created to test a more extensive dataset. It will take significantly longer to run than the other datasets.