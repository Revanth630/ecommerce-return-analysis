# E-Commerce Return Reduction Analysis

A Python version of my research on cutting product returns. I presented the original study at the **Alliance University International Conference**.

## Question
Which order attributes (category, discount, delivery time, payment mode, size-chart use) raise the chance that an order is returned?

## Method
- Descriptive analysis: return rate by category
- **Logistic regression**, read through **odds ratios** so the results are easy to explain to business users
- ROC-AUC to check how well the model separates returned from kept orders

## Findings this kind of analysis supports
- Apparel and footwear have the highest return rates. A **size-chart prompt** noticeably lowers returns
- Heavy discounts, slow delivery and cash-on-delivery all raise return odds
- Recommendations: push size guidance, nudge buyers toward prepaid payment, and set delivery SLAs for high-return categories

## Run
```bash
pip install -r requirements.txt
python return_analysis.py
```
> The dataset is synthetic. The original study was done in R and Python on transactional data.
