# 📈 AI Investment Analyst

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://investment-ai-agent-free-llm.streamlit.app/)

### *AI-powered stock comparison and investment analysis — fast, structured, and provider-agnostic.*

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-success)

---

## ⭐ Why this project?

Analyzing stocks requires:

* reliable market data
* structured financial reasoning
* objective comparison

**AI Investment Analyst** provides a **single, clean dashboard** to compare two stocks and generate **professional-grade analysis reports** using modern LLMs and live market data.

Built for **developers, students, and analysts** who want clarity without complexity.

---

## ✨ Key Features

* 📊 **Side-by-side stock comparison**
* 🧠 **LLM-powered financial reasoning**
* 🔁 **Multiple analysis modes**
* 🌐 **Cerebras & Groq support**
* 📈 **Live market data (Yahoo Finance)**
* 📥 **Markdown report export**
* 🎨 **Clean Streamlit UI**
* 🔐 **API keys never stored**

---

## 🖼 Screenshots

*(Add screenshots here if needed)*

---

## 🚀 Getting Started

### Requirements

* Python **3.9+**
* Internet connection
* API key from at least one provider:

  * Cerebras
  * Groq

---

### Install dependencies

```bash
pip install -r requirements.txt
```

---

### Run the app

```bash
streamlit run app.py
```

Open the local URL shown in your terminal.

---

## 🧪 How to Use

1. Select inference provider and model
2. Enter API key (via sidebar)
3. Enter two stock symbols (e.g., `AAPL`, `MSFT`)
4. Choose analysis type
5. Generate and download the report

---

## 📊 Analysis Modes

* Comprehensive Comparison
* Quick Overview
* Financial Metrics Focus
* Risk Analysis
* Technical Analysis

Each mode follows a **strict, structured output format**.

---

## 🔐 Security & Privacy

* API keys are used **in-memory only**
* No logging or storage
* No `.env` files required
* Direct API calls via OpenAI-compatible endpoints

---

## 🧠 Design Philosophy

* Clear over clever
* Structured over verbose
* Neutral, objective analysis
* No buy/sell recommendations
* Feels like a real internal tool

---

## 🛠 Built With

* **Python**
* **Streamlit**
* **Agno Agents**
* **Yahoo Finance**
* OpenAI-compatible REST APIs

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 💬 Final Note

This project is intentionally **focused and practical**.
It demonstrates how LLMs can assist with **real-world financial analysis** — without unnecessary complexity.

Happy building 🚀
