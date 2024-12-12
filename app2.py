import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools.tavily_search import TavilySearchResults
from fpdf import FPDF
import PyPDF2
import docx
import logging
from datetime import datetime
import os
import time

# Initialize LLM and search tools
llm = ChatGroq(api_key=st.secrets["groq_api_key"])
search = TavilySearchResults(api_key=st.secrets["tavily_api_key"], max_results=2)
parser = StrOutputParser()


def setup_session_logging():
    if 'logger' not in st.session_state:
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = "session_logs"
        os.makedirs(log_dir, exist_ok=True)
        logger = logging.getLogger(f"session_{session_id}")
        logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(f"{log_dir}/session_{session_id}.log")
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        st.session_state.logger = logger

def scrape_website(url):
    try:
        if not url.strip():
            st.session_state.logger.warning("No URL provided for scraping")
            return {"title": "No URL provided", "description": "No description available"}
        response = search.invoke(f"summarize content and key information from {url}")
        if response and len(response) > 0:
            content = response[0].get('content', 'No description available')
            title = response[0].get('title', 'No title available')
            st.session_state.logger.info(f"Successfully scraped website: {url}")
            return {"title": title, "description": content}
        st.session_state.logger.warning(f"No data found for URL: {url}")
        return {"title": "No data found", "description": "Could not fetch data from URL"}
    except Exception as e:
        st.error(f"Error accessing {url}: {str(e)}")
        st.session_state.logger.error(f"Error scraping website {url}: {str(e)}")
        return {"title": "Error", "description": f"Error scraping website: {str(e)}"}

def parse_uploaded_file(file):
    try:
        if file.type == "application/pdf":
            reader = PyPDF2.PdfReader(file)
            content = " ".join([page.extract_text() for page in reader.pages])
            st.session_state.logger.info(f"Successfully parsed PDF file: {file.name}")
            return content
        elif file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(file)
            content = " ".join([paragraph.text for paragraph in doc.paragraphs])
            st.session_state.logger.info(f"Successfully parsed DOCX file: {file.name}")
            return content
        st.session_state.logger.warning(f"Unsupported file type: {file.type}")
        return None
    except Exception as e:
        st.error(f"Error parsing file: {str(e)}")
        st.session_state.logger.error(f"Error parsing file {file.name}: {str(e)}")
        return None

def generate_insights(inputs, temperature, max_tokens):
    company_data = scrape_website(inputs["company_url"])
    competitors_data = []

    if inputs["competitors"]:
        competitors_data = [scrape_website(url.strip()) for url in inputs["competitors"].split(",") if url.strip()]

    company_title = company_data.get("title", "")
    company_description = company_data.get("description", "")

    prompt = f"""
    You are a seasoned sales assistant. Based on the following details:
    - **Company:** {{company_title}} ({{company_description}})
    - Product Name: {{product_name}}
    - Product Category: {{product_category}}
    - Competitors Data: {{competitors_data}}
    - Value Proposition: {{value_proposition}}
    - Target Customer: {{target_customer}}

    Generate insights focusing on:
    1. Company Strategy
    2. Competitor Mentions
    3. Leadership Information
    Provide actionable and concise information for the sales representative.
    """

    try:
        prompt_template = ChatPromptTemplate.from_template(prompt)
        chain = prompt_template | llm | parser
        insights = chain.invoke({
            "company_title": company_title,
            "company_description": company_description,
            "product_name": inputs['product_name'],
            "product_category": inputs['product_category'],
            "competitors_data": str(competitors_data),
            "value_proposition": inputs['value_proposition'],
            "target_customer": inputs['target_customer']
        })
        
        # Log the generated insights
        st.session_state.logger.info(f"Generated Insights:\n{insights}")
        
        return insights
    except Exception as e:
        st.error(f"Error generating insights: {str(e)}")
        st.session_state.logger.error(f"Error generating insights: {str(e)}")
        return None
    
def summarize_insights(company_insights):
    prompt = f"Summarize the following detailed content into a concise summary:\n\n{company_insights}"
    summarized_response = llm.invoke(prompt)
    return summarized_response

def generate_pdf(content, filename="Account_Insights.pdf"):
    try:
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        effective_page_width = pdf.w - 2*pdf.l_margin
        pdf.set_auto_page_break(auto=True, margin=15)
        
        # Add a title for the summary
        pdf.set_font('Arial', 'B', 16)
        pdf.cell(effective_page_width, 10, 'Sales Insights Summary', 0, 1, 'C')
        
        pdf.ln(10)  # Add a line space
        
        # Set the font back for content
        pdf.set_font("Arial", size=10)
        
        # Insert the content
        content = content.replace('\n', ' ')
        content = ' '.join(content.split())  # Ensure single spacing by removing extra spaces
        pdf.multi_cell(effective_page_width, 10, content)
        
        # Save PDF to a file
        pdf.output(filename)
        
        return filename
    except Exception as e:
        st.error(f"Error generating PDF: {str(e)}")
        return None

def main():
    setup_session_logging()

    # Initialize LLM and search tools
    global llm, search
    llm = ChatGroq(api_key=st.secrets["groq_api_key"])
    search = TavilySearchResults(api_key=st.secrets["tavily_api_key"], max_results=2)

    # Add a success message container that can be conditionally displayed
    reset_success_placeholder = st.empty()

    # Page Header
    st.title("Sales Assistant Agent")
    st.markdown("<h4 style='text-align: left; color: white;'>Assistant Agent Powered by Groq.</h4>", unsafe_allow_html=True)
    st.markdown('''
    This application generates insights for sales representatives based on product and company information.
    You can provide details such as the product name, company URL, product category, competitors, value proposition, and target customer.
    The application uses Groq's LLM model to generate insights focusing on company strategy, competitor mentions, and leadership information.
    ''')

    st.session_state.logger.info("Application started")

    # Add sliders for temperature and max tokens
    st.sidebar.title("LLM Settings")
    temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
    max_tokens = st.sidebar.slider("Max Tokens", min_value=100, max_value=2000, value=500, step=100)

    # Streamlit form
    with st.form("company_info", clear_on_submit=True):
        product_name = st.text_input("Product Name:")
        company_url = st.text_input("Company URL:")
        product_category = st.text_input("Product Category:")
        competitors = st.text_input("Competitors (comma-separated URLs):")
        value_proposition = st.text_area("Value Proposition:")
        target_customer = st.text_input("Target Customer:")

        uploaded_file = st.file_uploader("Upload Product Overview (optional):", type=["pdf", "docx"])

        col1, col2 = st.columns(2)
        with col1:
            if st.form_submit_button("Generate Insights"):
                if product_name and company_url:
                    st.session_state.logger.info(f"User submitted form with the following inputs:")
                    st.session_state.logger.info(f"Product Name: {product_name}")
                    st.session_state.logger.info(f"Company URL: {company_url}")
                    st.session_state.logger.info(f"Product Category: {product_category}")
                    st.session_state.logger.info(f"Competitors: {competitors}")
                    st.session_state.logger.info(f"Value Proposition: {value_proposition}")
                    st.session_state.logger.info(f"Target Customer: {target_customer}")
                    
                    with st.spinner("Processing..."):
                        inputs = {
                            "product_name": product_name,
                            "company_url": company_url,
                            "product_category": product_category,
                            "competitors": competitors,
                            "value_proposition": value_proposition,
                            "target_customer": target_customer,
                        }
                        if uploaded_file:
                            file_content = parse_uploaded_file(uploaded_file)
                            if file_content:
                                inputs["uploaded_file"] = file_content
                                st.session_state.logger.info(f"Uploaded file parsed: {uploaded_file.name}")

                        company_insights = generate_insights(inputs, temperature, max_tokens)
                        if company_insights:
                            st.session_state["company_insights"] = company_insights
                            st.session_state.logger.info("Insights generated and stored in session state")
                else:
                    st.warning("Please provide at least a product name and company URL.")
                    st.session_state.logger.warning("Incomplete form submission: missing product name or company URL")
        with col2:
            if st.form_submit_button("Reset Application"):
                st.session_state.clear()
                setup_session_logging()  # Reinitialize the logger
                
                # Display success message
                st.success("Application reset successfully! Ready for a new session.")
                st.session_state.logger.info("Application reset")
                
                time.sleep(3)  # Wait for 3 seconds
                st.rerun()  # Correct method to rerun the app

    # Display insights and download option
    if "company_insights" in st.session_state:
        st.subheader("Generated Insights")
        st.markdown(st.session_state["company_insights"])  # Ensure to convert newlines to spaces for single-spaced display

        summarized_insights = summarize_insights(company_insights)
        pdf_file = generate_pdf(summarized_insights)
        # pdf_file = generate_pdf(st.session_state["company_insights"])
        if pdf_file:
            with open(pdf_file, "rb") as pdf:
                st.download_button(
                    label="Download Insights as PDF",
                    data=pdf,
                    file_name=pdf_file,
                    mime="application/pdf"
                )
            st.session_state.logger.info("PDF generated and download button displayed")

    st.session_state.logger.info("Application session ended")

if __name__ == "__main__":
    main()
