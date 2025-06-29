# Credit Scoring Business Understanding

### 1. How does the Basel II Accord’s emphasis on risk measurement influence our need for an interpretable and well-documented model?

The Basel II Capital Accord establishes international standards for banks to quantify and manage credit risk to ensure financial stability. It mandates that financial institutions hold sufficient capital reserves proportional to their risk exposure. This regulatory framework requires that credit risk models be transparent, interpretable, and thoroughly documented to allow auditors, regulators, and stakeholders to understand the decision-making process behind credit assessments.

Interpretable models enhance trust, facilitate regulatory compliance, and enable banks to explain credit decisions to customers. According to the [Basel II overview], the emphasis on risk measurement necessitates models that not only predict accurately but also provide clear rationale for risk scoring, ensuring accountability and fairness in lending practices.

---

### 2. Since we lack a direct "default" label, why is creating a proxy variable necessary, and what are the potential business risks of making predictions based on this proxy?

In this challenge, the absence of a direct "default" indicator requires constructing a proxy target variable representing customer credit risk. This proxy is derived by analyzing behavioral data such as Recency, Frequency, and Monetary (RFM) transaction patterns to segment customers into risk categories. This approach aligns with alternative credit scoring methodologies used in fintech, as described in [HKMA's alternative credit scoring].

However, using a proxy introduces risks: it may imperfectly capture true default behavior, leading to misclassification. False positives may deny credit to creditworthy customers, harming customer experience and business growth, while false negatives increase the likelihood of loan defaults, impacting financial stability and capital requirements. Therefore, proxy-based predictions must be used cautiously and regularly validated against actual repayment outcomes.

---

### 3. What are the key trade-offs between using a simple, interpretable model (like Logistic Regression with WoE) versus a complex, high-performance model (like Gradient Boosting) in a regulated financial context?

**Simple, Interpretable Models (e.g., Logistic Regression with Weight of Evidence encoding):**

- *Advantages:*  
  - Transparency and ease of explanation facilitate regulatory approval and auditability ([ListenData WoE Explanation]).  
  - Easier maintenance and faster training times.  
  - Enables direct understanding of feature contributions to risk scores.

- *Disadvantages:*  
  - May have lower predictive accuracy when data relationships are complex or nonlinear.  
  - Limited ability to capture interactions without extensive manual feature engineering.

**Complex, High-Performance Models (e.g., Gradient Boosting Machines):**

- *Advantages:*  
  - Higher predictive power by capturing nonlinearities and complex feature interactions ([Analytics Vidhya Hyperparameter Tuning]).  
  - Potentially reduces financial losses by better identifying risky borrowers.

- *Disadvantages:*  
  - Models are often black-boxes, harder to interpret and explain to regulators and customers.  
  - Risk of regulatory pushback due to lack of transparency.  
  - Requires more computational resources and expertise to deploy and maintain.

In regulated finance, the trade-off is between *performance* and *interpretability*. While complex models can boost accuracy, simple models often better satisfy regulatory requirements and stakeholder trust. Hybrid approaches and explainability tools (e.g., SHAP values) can help bridge this gap but must be used with caution.
