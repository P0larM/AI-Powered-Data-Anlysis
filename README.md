
# AI-Powered Data Analyst

**An AI-powered statistical analysis platform combining data analytics, statistical methods, machine learning, and large language models.**

> 🚧 **WORK IN PROGRESS:** This project is actively under development. Features, architecture, and functionality are subject to change.

## Overview

AI-Powered Data Analyst is a personal project exploring how artificial intelligence can assist with real-world data analysis.

The goal is to develop a platform that allows users to explore datasets, perform statistical analyses, build predictive models, and interact with their data using natural language.

Rather than relying entirely on an LLM to interpret data, the platform integrates traditional statistical and machine learning libraries to perform computations, with AI helping users formulate questions and understand the results.

## Current Features

- **Dataset Loading:** Importing and working with structured datasets using Pandas.
- **Data Profiling:** Examining dataset dimensions, variable types, missing values, duplicate observations, descriptive statistics, and categorical distributions.
- **Correlation Analysis:** Pearson and Spearman correlation coefficients, associated p-values, and correlation matrices.
- **Categorical Analysis:** Chi-square tests of independence, expected frequency checks, and Cramér's V effect sizes.
- **Multiple Testing Correction:** Bonferroni and Benjamini-Hochberg false discovery rate adjustments.
- **Regression Analysis:** Linear regression with coefficient estimates, confidence intervals, p-values, R-squared, and adjusted R-squared.
- **AI Chatbot:** Natural language interaction with datasets, including AI-generated SQL queries and statistical analysis.

## Technologies

| Technology | Purpose |
|---|---|
| Python | Core development |
| Pandas & NumPy | Data manipulation and numerical computing |
| SciPy | Statistical testing |
| Statsmodels | Statistical modelling and regression |
| Scikit-learn | Machine learning |
| SQL | Data querying and management |
| Cohere API | LLM integration and natural language processing |
| Git & GitHub | Version control |

## Project Structure

```text
Ai-Data-Analyst/
│
├── app/
│   ├── data/
│   ├── statistical_analysis/
│   ├── machine_learning/
│   ├── database/
│   └── ai/
│
├── datasets/
├── tests/
├── requirements.txt
└── README.md
```

*The project structure is evolving as development continues.*

## Development Roadmap

- [x] Initial Python project setup
- [x] Dataset loading and profiling
- [x] Pearson and Spearman correlation analysis
- [x] Chi-square tests and Cramér's V
- [x] Multiple hypothesis testing corrections
- [x] Initial linear regression functionality
- [x] LLM-powered natural language data queries
- [ ] Expand regression methods, including logistic and regularized regression
- [ ] Add additional statistical tests and diagnostics
- [ ] Expand machine learning functionality and model evaluation
- [ ] Improve database integration and data management
- [ ] Develop an interactive user interface
- [ ] Expand automated testing and continuous integration
- [ ] Improve AI-generated statistical explanations and reporting

## Goals and Motivation

This project is an opportunity to combine my interests in statistics, computer science, data analytics, and artificial intelligence into one practical application.

A major focus is understanding how LLMs can complement traditional statistical methods rather than replace them. The goal is to build a tool that produces statistically grounded results while making those results easier to explore and interpret.

## Project Status

**Actively under development.**

This repository represents an ongoing learning and development project. New functionality, improvements, and architectural changes are being introduced as the project progresses.
