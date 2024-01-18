INFORMATION = {
    "Employee IDs": [
        39592,
        38836,
        37815,
        38433,
        42962,
        39976,
        42439,
        38076,
        34251,
        33479,
        39583,
        36145,
        42299,
        35888,
        38613,
        36370,
        33338,
        35048,
        37904,
        34556

    ]
}


leave_example_data = """
                            | Emp_ID | User_Name | Department | Leave_Type | Leaves_Available | Leaves_Availed | Leaves_Availed_Dates |
                            |---------|-----------|------------|------------|------------------|----------------|----------------------|
                            | 39592 | John Doe | HR | Annual | 12 | 5 | 2023-06-21, 2023-08-12, 2023-11-05, 2023-12-18, 2023-12-22 |
                            | 38836 | Jane Smith | Marketing | Sick | 10 | 3 | 2023-02-05, 2023-04-17, 2023-10-29 |
                            | 37815 | Robert Johnson | Finance | Maternity | 50 | 20 | 2023-01-20, 2023-01-21, 2023-01-22, 2023-01-23, 2023-01-24, 2023-01-25, 2023-01-26, 2023-01-27, 2023-01-28, 2023-01-29, 2023-01-30, 2023-01-31, 2023-02-01, 2023-02-02, 2023-02-03, 2023-02-04, 2023-02-05, 2023-02-06, 2023-02-07, 2023-02-08, 2023-02-09 |
                            | 38433 | Linda Williams | HR | Annual | 12 | 4 | 2023-03-15, 2023-05-23, 2023-09-08, 2023-11-21 |
                            | 42962 | Michael Brown | Marketing | Paternity | 15 | 5 | 2023-04-03, 2023-06-18, 2023-07-26, 2023-08-10, 2023-10-31 |
                            """

it_support_example_data = """
                    | Ticket_ID | Category | Subcategory | Emp_ID | User_Name | Assigned_To | Priority | Ticket_Status | Short_Issue_Description | Detailed_Issue_Description | Ticket_Opened_Time | Ticket_Resolved_Time |
                    |-----------|----------|-------------|---------|-----------|-------------|----------|---------------|------------------------|----------------------------|--------------------|----------------------|
                    | INC0010001| Software | Email | 39592 | John Doe | ITSupport1 | 2 - High | New | Cannot send email | User reports that they cannot send email from their account. Error message appears. | 2023-12-24 09:30 | |
                    | INC0010002| Hardware | Printer | 38836 | Jane Smith | ITSupport2 | 3 - Moderate | In Progress | Printer not working | User reports that the printer in their department is not working. | 2023-12-24 10:00 | |
                    | INC0010003| Software | CRM | 37815 | Robert Johnson | ITSupport3 | 1 - Critical | Resolved | CRM login issue | User reports that they cannot log into the CRM. Password reset did not resolve the issue. | 2023-12-24 11:00 | 2023-12-24 11:30 |
                    | INC0010004| Software | Email | 38433 | Linda Williams | ITSupport4 | 2 - High | New | Cannot send email | User reports that they cannot send email from their account. Error message appears. | 2023-12-24 09:30 | |
                    | INC0010005| Hardware | Printer | 42962 | Michael Brown | ITSupport5 | 3 - Moderate | In Progress | Printer not working | User reports that the printer in their department is not working. | 2023-12-24 10:00 | |
                    """

health_insurance_example_data = """
                    | Emp_ID | User_Name     | Policy_Name           | Policy_Status | Policy_End_Date | Sum_Insured (USD) | Available_Balance (USD) |
                    |---------|---------------|-----------------------|---------------|-----------------|-------------------|-------------------------|
                    | 39592   | John Doe      | Northwind Health Plus | Active        | 21-08-2024      | 5,00,000          | 4,50,000                |
                    | 38836   | Jane Smith    | Northwind Standard    | Lapsed        |                 | 2,50,000          | 0                       |
                    | 37815   | Robert Johnson| Northwind Health Plus | Active        | 15-11-2024      | 7,50,000          | 6,75,000                |
                    | 38433   | Linda Williams| Northwind Standard    | Lapsed        |                 | 3,00,000          | 0                       |
                    | 42962   | Michael Brown | Northwind Health Plus | Lapsed        |                 | 4,00,000          | 0                       |
                    """

