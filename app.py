# import necessary libraries
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
llm = ChatGroq(api_key=st.secrets["GROQ_API_KEY"])
search = TavilySearchResults(api_key=st.secrets["TAVILY_API_KEY"], max_results=2)
parser = StrOutputParser()

# define a function to set up session logging
def setup_session_logging():
    """
    _summary_: Set up logging for the current session. Creates a logger object and adds a file handler to log messages.
    """
    if 'logger' not in st.session_state:
        # Create a logger object for the current session
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = "session_logs"
        # Create the log directory if it does not exist
        os.makedirs(log_dir, exist_ok=True)
        logger = logging.getLogger(f"session_{session_id}")
        logger.setLevel(logging.INFO)
        # Create a file handler for the logger
        file_handler = logging.FileHandler(f"{log_dir}/session_{session_id}.log")
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        # Store the logger in the session state object
        st.session_state.logger = logger

# define the scrape_website function
def scrape_website(url):
    """_summary_: Scrape the content and key information from a website URL.
        _params_: url (str): The URL of the website to scrape.
    _returns_: dict: A dictionary containing the title and description of the website content.
    """

    try:
        # if the url is empty, return a default message and return a message
        if not url.strip():
            st.session_state.logger.warning("No URL provided for scraping")
            return {"title": "No URL provided", "description": "No description available"}
        response = search.invoke(f"summarize content and key information from {url}")
        # if the response is not empty and has content, extract the title and description from the response and return it 
        if response and len(response) > 0:
            content = response[0].get('content', 'No description available')
            title = response[0].get('title', 'No title available')
            st.session_state.logger.info(f"Successfully scraped website: {url}")
            return {"title": title, "description": content}
        # if no data is found for the URL, return a message
        st.session_state.logger.warning(f"No data found for URL: {url}")
        return {"title": "No data found", "description": "Could not fetch data from URL"}
    # if an exception occurs during the scraping process, log the error and return an error message
    except Exception as e:
        st.error(f"Error accessing {url}: {str(e)}")
        st.session_state.logger.error(f"Error scraping website {url}: {str(e)}")
        return {"title": "Error", "description": f"Error scraping website: {str(e)}"}

# define the parse_uploaded_file function
def parse_uploaded_file(file):
    """_summary_: Parse the content of an uploaded PDF or DOCX file.
        _params_: file (File): The uploaded file object.
    _returns_: str: The text content extracted from the file, or None if the file type is not supported.
    """
    try:
        # check the file type and extract the text content based on the file type
        # pdf files are parsed using PyPDF2, and docx files are parsed using the python-docx library
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
        # otherwise, log a warning for unsupported file types and return None
        st.session_state.logger.warning(f"Unsupported file type: {file.type}")
        return None
    # if an exception occurs during the parsing process, log the error and return None
    except Exception as e:
        st.error(f"Error parsing file: {str(e)}")
        st.session_state.logger.error(f"Error parsing file {file.name}: {str(e)}")
        return None

# define the generate_insights function
def generate_insights(inputs, temperature, max_tokens):
    """_summary_: Generate insights for sales representatives based on the provided inputs.
        _params_: inputs (dict): A dictionary containing the input data for generating insights.
        temperature (float): The temperature parameter for the LLM model.
        max_tokens (int): The maximum number of tokens to generate.
    _returns_: str: The generated insights as a formatted string.
    """
    # scrape the company website data and competitors data
    company_data = scrape_website(inputs["company_url"])
    competitors_data = []

    # if competitors are provided, scrape the data for each competitor URL
    if inputs["competitors"]:
        competitors_data = [scrape_website(url.strip()) for url in inputs["competitors"].split(",") if url.strip()]

    # generate insights based on the provided inputs and the scraped data
    company_title = company_data.get("title", "")
    company_description = company_data.get("description", "")

    # define the prompt template for generating insights
    prompt = f"""
    You are a seasoned sales assistant. Based on the following details:
    - **Company:** {{company_title}} ({{company_description}})
    - Product Name: {{product_name}}
    - Product Category: {{product_category}}
    - Competitors Data: {{competitors_data}}
    - Value Proposition: {{value_proposition}}
    - Target Customer: {{target_customer}}

    Product and Company Overview
    * Generate a concise summary of the product and company, including key features, benefits, and unique selling points.
    * Provide a recent news article or press release about the product or company to add current context.
    Competitive Landscape
    * compare the target product/product with the given competitors in terms of their offerings, highlighting strengths and weaknesses.
    * Analyze the product's value proposition and differentiation factors using a SWOT (Strengths, Weaknesses, Opportunities, Threats) framework.
    Target Customer Analysis
    * Define the ideal customer persona, including demographics, psychographics, and behavioral characteristics.
    * Identify key pain points of the target customer and explain how the product addresses each one.
    * Develop a unique selling proposition (USP) that clearly communicates the product's value to the target customer.
    Sales Strategy and Approach
    * Recommend an ideal sales approach (e.g., consultative selling, solution selling) and explain why it's suitable for this product and target customer.
    * Anticipate potential objections and provide concise, effective counterarguments for each.
    * Identify the most effective sales channels for reaching the target customer and explain the rationale for each.
    Sample Sales Pitch
    * Generate a sample sales pitch paragraph incorporating the insights from the above analysis, ensure the sales pitch is:
        - Attention-grabbing
        - Solution presentation
        - Product highlights and benefits
        - Call to action
    Social Media Content
    * Generate 2 sample social media posts that highlight key product benefits, address customer pain points, and include relevant hashtags.
    """

    try:
        # generate insights using the LLM model and the provided inputs and the scraped data
        prompt_template = ChatPromptTemplate.from_template(prompt)
        # define the chain of tools to be used for generating insights
        chain = prompt_template | llm | parser
        # invoke the chain with the provided inputs and scraped data
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
    # if an exception occurs during the generation process, log the error and return None
    except Exception as e:
        st.error(f"Error generating insights: {str(e)}")
        st.session_state.logger.error(f"Error generating insights: {str(e)}")
        return None

# define the generate_pdf function
def generate_pdf(content, filename="Account_Insights.pdf"):
    """_summary_: Generate a PDF file with the provided content.
        _params_: content (str): The content to be included in the PDF file.
        filename (str): The filename for the generated PDF file.
    _returns_: str: The filename of the generated PDF file, or None if an error occurs.
    """
    try:
        # create a PDF file with the provided content
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=10)
        # set the space between lines = 10
        pdf.multi_cell(0, 10, content)
        pdf.output(filename)
        st.session_state.logger.info(f"Successfully generated PDF: {filename}")
        return filename
    except Exception as e:
        st.error(f"Error generating PDF: {str(e)}")
        st.session_state.logger.error(f"Error generating PDF: {str(e)}")
        return None

# define the main function to be called when the page is loaded and run the application
def main():
    # Set up session logging
    setup_session_logging()

    # Initialize LLM and search tools
    global llm, search
    llm = ChatGroq(api_key=st.secrets["GROQ_API_KEY"])
    search = TavilySearchResults(api_key=st.secrets["TAVILY_API_KEY"], max_results=2)

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

    # add a space before the application starts logging
    st.markdown("<br>", unsafe_allow_html=True)
    st.session_state.logger.info("Application started")

    # Add sliders for temperature and max tokens
    st.sidebar.title("LLM Settings")
    temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
    max_tokens = st.sidebar.slider("Max Tokens", min_value=100, max_value=2000, value=500, step=100)

    # Streamlit form
    with st.form("company_info", clear_on_submit=True):
        # Form fields
        product_name = st.text_input("Product Name:")
        company_url = st.text_input("Company URL:")
        product_category = st.text_input("Product Category:")
        competitors = st.text_input("Competitors (comma-separated URLs):")
        value_proposition = st.text_area("Value Proposition:")
        target_customer = st.text_input("Target Customer:")
        # Upload file
        uploaded_file = st.file_uploader("Upload Product Overview (optional):", type=["pdf", "docx"])

        # Form columns for buttons and layout
        col1, col2 = st.columns(2)
        with col1:
            # Generate Insights button
            if st.form_submit_button("Generate Insights"):
                if product_name and company_url:
                    # if the form is submitted with the required fields, log the form inputs
                    st.session_state.logger.info(f"User submitted form with the following inputs:")
                    st.session_state.logger.info(f"Product Name: {product_name}")
                    st.session_state.logger.info(f"Company URL: {company_url}")
                    st.session_state.logger.info(f"Product Category: {product_category}")
                    st.session_state.logger.info(f"Competitors: {competitors}")
                    st.session_state.logger.info(f"Value Proposition: {value_proposition}")
                    st.session_state.logger.info(f"Target Customer: {target_customer}")
                    
                    # Generate insights based on the form inputs, include a spinner to show the results is being generated
                    with st.spinner("Processing..."):
                        inputs = {
                            "product_name": product_name,
                            "company_url": company_url,
                            "product_category": product_category,
                            "competitors": competitors,
                            "value_proposition": value_proposition,
                            "target_customer": target_customer,
                        }
                        # if an uploaded file is provided, parse the content of the file
                        if uploaded_file:
                            file_content = parse_uploaded_file(uploaded_file)
                            if file_content:
                                inputs["uploaded_file"] = file_content
                                st.session_state.logger.info(f"Uploaded file parsed: {uploaded_file.name}")
                        # generate insights based on the provided inputs
                        company_insights = generate_insights(inputs, temperature, max_tokens)
                        if company_insights:
                            st.session_state["company_insights"] = company_insights
                            st.session_state.logger.info("Insights generated and stored in session state")
                else:
                    # if the form is submitted without the required fields, show a warning message
                    st.warning("Please provide at least a product name and company URL.")
                    st.session_state.logger.warning("Incomplete form submission: missing product name or company URL")
        # Reset Application button
        with col2:
            if st.form_submit_button("Reset Application"):
                st.session_state.clear()
                setup_session_logging()  # Reinitialize the logger
                
                # Display success message
                reset_success_placeholder.success("Application reset successfully! Ready for a new session.")
                st.session_state.logger.info("Application reset")
                
                time.sleep(3)  # Wait for 3 seconds
                st.rerun()

    # Display insights and download option
    if "company_insights" in st.session_state:
        st.subheader("Generated Insights")
        st.markdown(st.session_state["company_insights"])

        # Add a download button for the insights as a PDF file
        pdf_file = generate_pdf(st.session_state["company_insights"])
        if pdf_file:
            with open(pdf_file, "rb") as pdf:
                st.download_button(
                    label="Download Insights as PDF",
                    data=pdf,
                    # set the filename to include the product name and the date and time of download
                    file_name=f"{product_name}_Insights_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                )
            st.session_state.logger.info("PDF generated and download button displayed")
    st.session_state.logger.info("Application session ended")
    # insert a space after the session end
    st.markdown("<br>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()  # Run the main function