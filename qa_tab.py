import streamlit as st
from openai import OpenAI
import inspect, re
from pathlib import Path

# ─── Initialize OpenAI Client ─────────────────────────────────────────────

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"].strip())



# ─── Ask LLM (OpenAI GPT-4o-mini) ─────────────────────────────────────────
def ask_llm(question: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant that explains Python code."},
            {"role": "user", "content": question},
        ],
    )
    return resp.choices[0].message.content

# ─── Extract Code Block Based on Label ────────────────────────────────────
def extract_code_block(label: str, file_path: Path) -> str:
    """
    Extract code from app.py between markers like:
    ##### WACC START ##### ... ##### WACC END #####
    """
    escaped = re.escape(label.upper())
    pattern = re.compile(
    rf"^\s*##### {escaped} START #####\s*$" +
    r"(.*?)" +
    rf"^\s*##### {escaped} END #####\s*$",
    flags=re.MULTILINE | re.DOTALL,
)

    try:
        text = file_path.read_text()
        match = pattern.search(text)
        return match.group(1).strip() if match else None
    except Exception as e:
        return f"#⚠️ Could not read file: {e}"

# ─── Render Q&A Tab ───────────────────────────────────────────────────────
def render_qa_tab():
    st.markdown("### ❓ Q&A")
    question = st.text_area("Ask me anything…", height=150,
                            placeholder="Enter a question, e.g. How did you calculate your WACC? Include acronyms i.e. FCFE, or LLM to get code snippets as well")

    if st.button("Run"):
        q = question.strip()
        if not q:
            st.warning("Please type a question.")
            return

        q_lower = q.lower()
        app_dir = Path(__file__).parent

        files_to_search = {
            "APP": app_dir / "app.py",
            "FCF": app_dir / "fcf_calculations.py"
        }

        # 🔑 Map keywords to block labels
        code_keywords = {
            "wacc": "WACC",
            "fama french": "FAMA FRENCH",
            "FF5": "FAMA FRENCH",
            "capm": "CAPM",
            "irr": "IRR",
            "discounted cash flow": "DCF",
            "terminal value": "TV",
            "nwc": "NWC",
            "cost of equity": "COE",
            "cost of debt": "COD",
            "fcfe": "FCFE",
            "fcff": "FCFF",
            "damo": "DAMODARAN",
            "betas": "BETAS",
            "evebitda": "EV/EBITDA",
            "ev/ebitda": "EV/EBITDA",
            "llm": "LLM"
        }

        shown_block = False
        for trigger, label in code_keywords.items():
            if trigger in q_lower:
                for file_label, file_path in files_to_search.items():
                    snippet = extract_code_block(label, file_path)
                    if snippet:
                        st.markdown(f"### 📄 Source code for `{label}` block (from `{file_label}`):")
                        st.code(snippet, language="python")
                        shown_block = True
                        break
            if shown_block:
                break

        with st.spinner("Querying LLM…"):
            answer = ask_llm(q)
        st.markdown("**Answer:**")
        st.write(answer)

        # Optional: show the source of ask_llm()
        st.markdown("**Helper function source (`ask_llm`)::**")
        st.code(inspect.getsource(ask_llm), language="python")
