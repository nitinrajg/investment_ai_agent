import streamlit as st
from agno.agent import Agent
from agno.run.agent import RunOutput
from agno.models.openai.like import OpenAILike
from agno.tools.yfinance import YFinanceTools

st.set_page_config(
    page_title="AI Investment Analyst",
    page_icon="📈",
    layout="wide"
)

st.title("AI Investment Analyst 📈🤖")
st.caption("Compare stock performance and generate detailed investment analysis reports.")

# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")
    
    provider = st.selectbox(
        "Inference Provider",
        options=["Cerebras", "Groq"],
        index=0,
        help="Choose the API provider for the LLM"
    )
    
    if provider == "Cerebras":
        api_key = st.text_input("Cerebras API Key", type="password", help="Enter your Cerebras Inference API key")
        model_options = {
            "Llama 3.3 70B": "llama-3.3-70b",
            "Llama 3.1 70B": "llama3.1-70b",
            "Llama 3.1 8B": "llama3.1-8b",
            # Add more Cerebras models here if desired
        }
        base_url = "https://api.cerebras.ai/v1"
    else:  # Groq
        api_key = st.text_input("Groq API Key", type="password", help="Enter your Groq API key")
        model_options = {
            "Llama 3.3 70B Versatile": "llama-3.3-70b-versatile",
            "Llama 3.1 70B": "llama3-70b-8192",
            "Llama 3.1 8B": "llama3-8b-8192",
            "Mixtral 8x7B": "mixtral-8x7b-32768",
            "Gemma 7B": "gemma-7b-it",
            # Add more Groq models here if desired
        }
        base_url = "https://api.groq.com/openai/v1"
    
    selected_model_name = st.selectbox("Model", options=list(model_options.keys()))
    model_id = model_options[selected_model_name]
    
    if api_key:
        st.success(f"✅ {provider} API Key configured")
    else:
        st.warning("⚠️ Please enter your API key to continue")

if api_key:
    # Create the agent dynamically based on provider and model
    assistant = Agent(
        model=OpenAILike(
            id=model_id,
            api_key=api_key,
            base_url=base_url
        ),
        tools=[YFinanceTools()],
        description=(
            "Senior equity research analyst with expertise in fundamental analysis, "
            "financial modeling, technical indicators, and market sentiment."
        ),
        instructions=[
            "Use Markdown. Headings (##, ###), tables for numbers, bullets for insights.",
            "Use emojis sparingly: 📊 data, 📈 trends, ⚠️ risks.",
            "Always explain what each metric means and why it matters.",
            "Include data source and timestamp for all financial data.",
            "Compare metrics vs historical averages or industry benchmarks when possible.",
            "Present both bullish and bearish signals objectively.",
            "Currency: $1,234.56 | Percentages: 12.34% | Dates: YYYY-MM-DD",
            "Always include units (Market Cap $B, P/E as x).",
            "Include a non-financial-advice disclaimer in every response.",
            "Explicitly flag missing, stale, or limited data.",
            "If analyst opinions conflict, show multiple viewpoints.",
            "If ticker is invalid, suggest similar symbols or ask for clarification.",
            "If real-time data is unavailable, state that delayed/historical data is used.",
            "Break multi-part requests into clear sections.",
            "If data fetch fails, explain the failure and suggest alternatives.",
            "Suggest 1–2 relevant follow-up analyses when useful.",
            """When analyzing a stock, always use this structure:
            1. Executive Summary (2–3 bullets)
            2. Current Price & Performance
            3. Fundamental Metrics
            4. Analyst Consensus
            5. Risk Factors
            6. Conclusion (no buy/sell advice)
            """,
            "Ask clarifying questions only if the request is ambiguous.",
            "Avoid unnecessary jargon.",
            "Show formulas briefly when calculations are involved."
        ],
        markdown=True,
    )

    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        stock1 = st.text_input(
            "First Stock Symbol",
            placeholder="e.g., AAPL",
            help="Enter the ticker symbol of the first stock"
        ).upper()
    with col2:
        stock2 = st.text_input(
            "Second Stock Symbol",
            placeholder="e.g., MSFT",
            help="Enter the ticker symbol of the second stock"
        ).upper()
    
    analysis_type = st.selectbox(
        "Analysis Type",
        [
            "Comprehensive Comparison",
            "Quick Overview",
            "Financial Metrics Focus",
            "Risk Analysis",
            "Technical Analysis"
        ]
    )
    
    if stock1 and stock2:
        if st.button("🔍 Generate Analysis", type="primary", use_container_width=True):
            with st.spinner(f"Analyzing {stock1} vs {stock2} using {provider} ({selected_model_name})..."):
                try:
                    query_templates = {
                        "Comprehensive Comparison": (
                            f"Conduct a full investment-grade comparison of {stock1} and {stock2}.\n\n"
                            "Required sections (use all, in this order):\n"
                            "1. Executive Summary\n"
                            "2. Price & Performance Comparison\n"
                            "3. Fundamental Metrics Comparison\n"
                            "4. Analyst Consensus Comparison\n"
                            "5. Risk Factors Comparison\n"
                            "6. Conclusion\n\n"
                            "Guidelines:\n"
                            "- Use side-by-side tables wherever possible\n"
                            "- Highlight relative strengths and weaknesses\n"
                            "- Maintain a neutral, long-term investor perspective\n"
                            "- Do NOT give buy/sell recommendations"
                        ),
                        "Quick Overview": (
                            f"Provide a high-level snapshot comparison of {stock1} and {stock2}.\n\n"
                            "Only include:\n"
                            "- Current price\n"
                            "- Market capitalization\n"
                            "- P/E ratio\n"
                            "- Recent price performance\n\n"
                            "Rules:\n"
                            "- Use ONE compact comparison table\n"
                            "- Follow with at most 3 bullet points summarizing key differences\n"
                            "- Do NOT include risk analysis, analyst opinions, or deep fundamentals"
                        ),
                        "Financial Metrics Focus": (
                            f"Compare {stock1} and {stock2} strictly from a financial fundamentals perspective.\n\n"
                            "Focus ONLY on:\n"
                            "- Profitability (gross, operating, net margins, ROE, ROA)\n"
                            "- Valuation (P/E, P/B, EV/EBITDA if available)\n"
                            "- Financial health (debt levels, liquidity, cash flow strength)\n\n"
                            "Rules:\n"
                            "- Ignore stock price movement and technical indicators\n"
                            "- Ignore analyst ratings\n"
                            "- Explain what each metric indicates about business quality\n"
                            "- Use tables first, explanation second"
                        ),
                        "Risk Analysis": (
                            f"Compare risk profiles of {stock1} vs {stock2}.\n\n"
                            "Focus on:\n"
                            "- Beta and volatility\n"
                            "- 52-week price range\n"
                            "- Debt levels\n"
                            "- Revenue/earnings stability\n\n"
                            "Format: Brief table + 3-4 key risk points. State which is riskier."
                        ),
                        "Technical Analysis": (
                            f"Compare technical profiles of {stock1} vs {stock2}.\n\n"
                            "Include:\n"
                            "- Current trend (bullish/bearish/neutral)\n"
                            "- Moving averages (50-day, 200-day)\n"
                            "- RSI if available\n"
                            "- Volume trends\n\n"
                            "Format: Compact table + momentum comparison (2-3 sentences)."
                        )
                    }
                    
                    query = query_templates.get(analysis_type, query_templates["Comprehensive Comparison"])
                    
                    response: RunOutput = assistant.run(query, stream=False)
                    
                    st.markdown("---")
                    st.markdown(response.content)
                    
                    st.divider()
                    col_export1, col_export2 = st.columns([3, 1])
                    with col_export2:
                        st.download_button(
                            label="📥 Download Report",
                            data=response.content,
                            file_name=f"{stock1}_vs_{stock2}_analysis.md",
                            mime="text/markdown"
                        )
                
                except Exception as e:
                    st.error(f"❌ Error during analysis: {str(e)}")
                    st.info("Please check that the stock symbols are valid and try again.")
    
    elif stock1 or stock2:
        st.info("👆 Please enter both stock symbols to begin the analysis.")
else:
    st.info("👈 Please select a provider and enter your API key in the sidebar to get started.")
    
    with st.expander("ℹ️ How to get API Keys"):
        st.markdown("""
        - **Cerebras**: Go to [https://cloud.cerebras.ai](https://cloud.cerebras.ai) → Sign up / log in → API Keys → Create new key
        - **Groq**: Go to [https://console.groq.com/keys](https://console.groq.com/keys) → Create new API key (free tier available with high rate limits)
        """)
    
    with st.expander("📊 Features"):
        st.markdown("""
        - **Multiple Providers**: Switch between Cerebras and Groq inference
        - **Model Selection**: Choose from popular models for each provider
        - **Comprehensive Stock Comparison**: Compare two stocks side-by-side
        - **Multiple Analysis Types**: Choose from 5 different analysis perspectives
        - **Real-time Data**: Fetches latest market data via YFinance
        - **Professional Reports**: AI-generated detailed investment analysis
        - **Export Capability**: Download reports as markdown files
        """)

st.divider()
st.caption("⚠️ Disclaimer: This tool provides information for educational purposes only and is not financial advice. Always consult with a licensed financial advisor before making investment decisions.")