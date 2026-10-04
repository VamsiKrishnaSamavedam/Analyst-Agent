# Messy retail orders test dataset

Upload `messy_retail_orders.csv` through the Analyst Desk interface.

It is intentionally unsuitable for direct reporting and includes:

- An exact duplicate row (`ORD-1002`) and a probable duplicate customer/order (`ORD-1026`).
- Leading/trailing spaces in names, regions, products, and statuses.
- Inconsistent region and status categories, including casing and spelling variants.
- Missing email, phone, price, discount, total, quantity, and status values.
- Likely PII columns (`customer_name`, `email`, and `phone`) that should be profiled but whose sample values should not be given to Gemini.
- Mixed numeric representations: currency symbols, percentage text, and a word instead of a number.
- Inconsistent and invalid dates, including an impossible month.
- Data-integrity problems: zero/negative quantity, negative amount, and an extreme quantity outlier (`9999`).
- A few totals that should be recalculated only after a business rule is explicitly defined.

Expected safe first-pass actions are duplicate removal and text trimming. Treat category normalization, date/type conversion, missing-value treatment, outlier treatment, and total reconciliation as review decisions rather than automatic fixes.
