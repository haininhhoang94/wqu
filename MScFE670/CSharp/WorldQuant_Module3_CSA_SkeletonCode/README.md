# Submission 1: Using C# and Excel to Track Property Prices

We are already provided a skeleton code.

<b>Purpose:</b>

- Implement statistical method to gain further insight into the overall trends
  of the property market

<b>Tasks:</b>

1. Set up the worksheet when the application is launched for the first time -
   The main method already calls a method "SetUp"; therefore, you simply have
   to implement this method, which should create a new Excel workbook titled
   "property_pricing.xlsx"

2. Implement the adding of property information to the sheet - The property
   information headers are as follows:

   a. Size (in square feet)

   b. Suburb

   c. City

   d. Market value

   The command line interface already calls a method "AddPropertyToWorksheet",
   so you simply have to implement this method

3. Implement statistical methods - In the skeleton code you find the following
   four statistical methods already declared:

   a. Mean market value

   b. Variance in market value

   c. Minimum market value

   d. Maximum market value

   Your task is to update these methods to perform the correct work based on
   the data from the sheet. These methods are already called in the command
   line interface, so you only need to implement the method.
